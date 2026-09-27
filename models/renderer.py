"""Differentiable, depth-sorted point splatting in PyTorch3D NDC coordinates."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class RenderConfig:
    render_size: int = 256
    points_per_pixel: int = 128
    radius: float = 4.0  # pixels
    weight_calculation: str = "exponential"
    compositor: str = "alpha"
    backend: str = "pytorch3d"
    sigma: float = 2.0  # Gaussian standard deviation in pixels (implementation choice)
    eta: float = 1.0
    background: float = 0.0
    pixel_chunk_size: int = 256


def linear_alpha(dist_xy, radius):
    return 1 - dist_xy.permute(0, 3, 1, 2) / (radius * radius)


def exponential_alpha(dist_xy, radius):
    return torch.exp(-dist_xy.permute(0, 3, 1, 2).clamp_min(0) / (radius * radius))


class PointRenderer(nn.Module):
    """Render B x N x 3 points with B x N x C features.

    ``pytorch3d`` is the production rasterizer. ``torch`` is a slower reference
    rasterizer for small CPU tests; it uses the same disk, depth and blending
    rules, with memory bounded by a configurable pixel chunk size.
    """

    def __init__(self, render_cfg=None):
        super().__init__()
        cfg = render_cfg or RenderConfig()
        size = cfg.render_size
        self.image_size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("render_size must be positive or an (H, W) pair")
        self.S = size
        self.K = cfg.points_per_pixel
        self.r = 2 * cfg.radius / min(self.image_size)
        self.sigma = 2 * getattr(cfg, "sigma", cfg.radius / math.sqrt(2)) / min(self.image_size)
        self.eta = getattr(cfg, "eta", 1.0)
        self.backend = getattr(cfg, "backend", "pytorch3d")
        self.weight_calculation = cfg.weight_calculation
        self.compositor_name = cfg.compositor
        self.background = getattr(cfg, "background", 0.0)
        self.chunk_size = getattr(cfg, "pixel_chunk_size", 256)
        if (self.K < 1 or self.chunk_size < 1
                or any(not math.isfinite(v) or v <= 0 for v in (self.r, self.sigma, self.eta))):
            raise ValueError("Rasterization settings must be positive")
        if not math.isfinite(self.background) or not 0 <= self.background <= 1:
            raise ValueError("background must be a finite RGB value in [0, 1]")
        if self.weight_calculation not in ("linear", "exponential"):
            raise ValueError("Unknown splat weight calculation")
        if self.compositor_name not in ("alpha", "weighted_sum", "norm_weighted_sum"):
            raise ValueError("Unknown compositor")
        if self.backend == "pytorch3d":
            try:
                from pytorch3d.renderer import compositing
                from pytorch3d.renderer.points import rasterize_points
                from pytorch3d.structures import Pointclouds
            except ImportError as exc:
                raise ImportError("Install PyTorch3D for training, or use renderer backend 'torch' for small tests") from exc
            self._rasterize = rasterize_points
            self._pointclouds = Pointclouds
            self._compositor = {
                "alpha": compositing.alpha_composite,
                "weighted_sum": compositing.weighted_sum,
                "norm_weighted_sum": compositing.norm_weighted_sum,
            }[self.compositor_name]
        elif self.backend != "torch":
            raise ValueError("Renderer backend must be 'pytorch3d' or 'torch'")

    def _weights(self, distance, valid):
        if self.weight_calculation == "linear":
            weights = 1 - distance / self.r ** 2
        else:
            weights = torch.exp(-distance.clamp_min(0) / (2 * self.sigma ** 2))
        weights = weights.clamp(0, 0.99)
        # Fractional eta must not differentiate pow(0, eta): its infinite
        # derivative can produce NaNs even for subsequently masked fragments.
        positive = valid & (weights > 0)
        return weights.clamp_min(torch.finfo(weights.dtype).tiny).pow(self.eta) * positive

    def _composite(self, features, idx, distances, depths):
        # idx is local to each batch, shape B x P x K; -1 marks empty slots.
        valid = idx >= 0
        weights = self._weights(distances, valid)
        batch_ids = torch.arange(features.shape[0], device=features.device)[:, None, None]
        colors = features[batch_ids, idx.clamp_min(0)]
        if self.compositor_name == "alpha":
            prefix = torch.cat((torch.ones_like(weights[..., :1]), 1 - weights), dim=-1)
            transmittance = prefix.cumprod(-1)
            blend = weights * transmittance[..., :-1]
            output = (blend.unsqueeze(-1) * colors).sum(-2)
            output = output + transmittance[..., -1:] * self.background
        elif self.compositor_name == "norm_weighted_sum":
            blend = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            output = (blend.unsqueeze(-1) * colors).sum(-2)
            output = torch.where(valid.any(-1, keepdim=True), output, output.new_tensor(self.background))
        else:
            output = (weights.unsqueeze(-1) * colors).sum(-2)
        depth_weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        depth = (depth_weights * depths.masked_fill(~valid, 0)).sum(-1)
        return output, depth, valid.any(-1), weights

    def _torch_fragments(self, points, pixels):
        distance = (points[:, None, :, :2] - pixels[None, :, None, :]).square().sum(-1)
        depth = points[:, None, :, 2].expand_as(distance)
        valid = (distance <= self.r ** 2) & (depth > 0)
        depths, indices = depth.masked_fill(~valid, float("inf")).topk(
            min(self.K, points.shape[1]), dim=-1, largest=False, sorted=True)
        selected_distances = distance.gather(-1, indices)
        indices = indices.masked_fill(~torch.isfinite(depths), -1)
        return indices, selected_distances, depths

    def _render_pytorch3d(self, points, features, return_raster):
        """Use fused compositing to avoid a B x H x W x K x C color tensor."""
        # PyTorch3D clips negative depth; also exclude exactly zero depth to
        # match the reference renderer and the positive-Z pinhole convention.
        depth = torch.where(points[..., 2:] > 0, points[..., 2:], -torch.ones_like(points[..., 2:]))
        raster_points = torch.cat((points[..., :2], depth), dim=-1)
        pcd = self._pointclouds(points=raster_points, features=features)
        idx, zbuf, distance = self._rasterize(pcd, self.S, self.r, self.K)
        valid = idx >= 0
        weights = self._weights(distance, valid).permute(0, 3, 1, 2).contiguous()
        packed_idx = idx.permute(0, 3, 1, 2).long().contiguous()
        output = self._compositor(packed_idx, weights, pcd.features_packed().T.contiguous())
        if self.compositor_name == "alpha":
            output = output + (1 - weights).prod(dim=1, keepdim=True) * self.background
        elif self.compositor_name == "norm_weighted_sum":
            output = torch.where(valid.any(-1).unsqueeze(1), output, output.new_tensor(self.background))
        depth_weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        depth = (depth_weights * zbuf.permute(0, 3, 1, 2).masked_fill(~valid.permute(0, 3, 1, 2), 0)).sum(1, keepdim=True)
        mask = valid.any(-1).to(points.dtype)
        result = {
            "feats": output, "depth": depth, "mask": mask,
            "valid_rays": mask.mean((1, 2)), "valid_pts": valid.to(points.dtype).mean((1, 2, 3)),
        }
        if return_raster:
            offset = torch.arange(points.shape[0], device=points.device)[:, None, None, None] * points.shape[1]
            local_idx = torch.where(idx >= 0, idx - offset, -1)
            result["raster_output"] = {
                "idx": local_idx.permute(0, 3, 1, 2), "zbuf": zbuf,
                "dist_xy": distance, "alphas": weights, "points": points, "feats": features,
            }
        return result

    def forward(self, points, features, return_raster=True):
        if (points.ndim != 3 or points.shape[-1] != 3 or features.ndim != 3
                or points.shape[:2] != features.shape[:2] or points.shape[1] == 0):
            raise ValueError("Expected points (B, N, 3) and features (B, N, C)")
        if self.backend == "pytorch3d":
            return self._render_pytorch3d(points, features, return_raster)
        batch, count, _ = points.shape
        height, width = self.image_size
        outputs, depths, masks = [], [], []
        indices_out, distances_out, zbuf_out, weights_out = [], [], [], []
        valid_counts = points.new_zeros(batch)
        y, x = torch.meshgrid(
            torch.arange(height, device=points.device, dtype=points.dtype),
            torch.arange(width, device=points.device, dtype=points.dtype), indexing="ij")
        pixels = torch.stack(((width - 1) / 2 - x, (height - 1) / 2 - y), -1)
        pixels = pixels.reshape(-1, 2) * (2.0 / min(height, width))
        fragments = (self._torch_fragments(points, chunk) for chunk in pixels.split(self.chunk_size))
        for idx, dist, zbuf in fragments:
            output, depth, mask, weights = self._composite(features, idx, dist, zbuf)
            outputs.append(output)
            depths.append(depth)
            masks.append(mask)
            valid_counts += (idx >= 0).sum((1, 2))
            if return_raster:
                padding = (0, self.K - idx.shape[-1])
                indices_out.append(torch.nn.functional.pad(idx, padding, value=-1))
                distances_out.append(torch.nn.functional.pad(dist.masked_fill(idx < 0, -1), padding, value=-1))
                zbuf_out.append(torch.nn.functional.pad(zbuf.masked_fill(idx < 0, -1), padding, value=-1))
                weights_out.append(torch.nn.functional.pad(weights, padding))
        # These diagnostics preserve the original renderer's dictionary interface.
        mask = torch.cat(masks, 1).reshape(batch, height, width).to(points.dtype)
        result = {
            "feats": torch.cat(outputs, 1).transpose(1, 2).reshape(batch, -1, height, width),
            "depth": torch.cat(depths, 1).reshape(batch, 1, height, width),
            "mask": mask,
            "valid_rays": mask.mean((1, 2)),
            "valid_pts": valid_counts / (height * width * self.K),
        }
        if return_raster:
            idx = torch.cat(indices_out, 1).reshape(batch, height, width, -1)
            result["raster_output"] = {
                "idx": idx.permute(0, 3, 1, 2),
                "zbuf": torch.cat(zbuf_out, 1).reshape(batch, height, width, -1),
                "dist_xy": torch.cat(distances_out, 1).reshape(batch, height, width, -1),
                "alphas": torch.cat(weights_out, 1).reshape(batch, height, width, -1).permute(0, 3, 1, 2),
                "points": points, "feats": features,
            }
        return result
