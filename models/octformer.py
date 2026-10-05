"""OctFormer on batched dense tensors, without an octree library.

Re-implements OctFormer (Wang, "OctFormer: Octree-based Transformers for 3D
Point Clouds", SIGGRAPH 2023; https://github.com/octree-nn/octformer, MIT
license) for batches of equally sized point clouds:

- octree node order: points sorted by the z-order (octree) or Hilbert key of
  their voxel, recomputed at every level;
- octree patch partition: B x N x C tensors reshaped into windows of
  ``patch_size`` consecutive points, or of points ``dilation`` apart in
  dilated blocks; padding is masked, and samples never share a window.
  Stages from ``full_attention_from`` on, which hold few points after
  downsampling, attend over the whole cloud instead;
- octree depthwise convolution (CPE) and 3 x 3 x 3 convolutions: a kernel
  over the 27 voxel offsets, applied to each point's k nearest neighbors
  (neighbors in the same offset bin are averaged);
- stride-2 octree convolution: pooling ``stride`` consecutive points along
  the curve, which keeps every sample at the same size;
- octree upsampling in the FPN head: the exact inverse of that pooling.

The voxel size doubles at every level. ``forward`` returns DGCNN's interface:
max-pooled global features, per-point features in input order and, with
``return_levels``, the FPN level features at full resolution.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.common import square_distance
from models.serialization import CURVES, curve_key

MASKED = -1e4  # Finite, like OctFormer's -1e3: fully padded windows stay finite.


def gather_points(x, index):
    """``x[b, index[b, ...]]`` for B x N x C ``x``. torch.gather's backward adds
    gradients atomically, far faster than advanced indexing's sorted accumulation."""
    flat = index.reshape(len(x), -1, 1).expand(-1, -1, x.shape[-1])
    return x.gather(1, flat).view(*index.shape, x.shape[-1])


class Level:
    """Points of one level in curve order, with their voxel grid and neighborhoods."""

    def __init__(self, xyz, voxel, curve, k, refine_bits=6):
        origin = xyz.amin(1, keepdim=True)
        fine = torch.floor((xyz - origin) / (voxel / 2 ** refine_bits)).long()
        bits = max(1, int(fine.max().item()).bit_length())
        # A fine key orders cells exactly like the voxel key and breaks ties spatially.
        self.order = curve_key(fine, bits, curve).argsort(dim=1)
        self.xyz = gather_points(xyz, self.order)
        self.grid = torch.floor((self.xyz - origin) / voxel).long()
        self.voxel = voxel
        distance, neighbors = square_distance(self.xyz, self.xyz).topk(min(k, xyz.shape[1]), dim=-1, largest=False)
        self.neighbors = neighbors
        offset = torch.round((gather_points(self.xyz, neighbors) - self.xyz.unsqueeze(2)) / voxel).clamp(-1, 1).long() + 1
        self.cells = offset[..., 0] * 9 + offset[..., 1] * 3 + offset[..., 2]
        counts = torch.zeros(*self.cells.shape[:2], 27, device=xyz.device).scatter_add_(
            2, self.cells, torch.ones_like(self.cells, dtype=xyz.dtype))
        self.share = 1 / counts.gather(2, self.cells)  # Average neighbors that share an offset bin.


class VoxelConv(nn.Module):
    """3 x 3 x 3 convolution over binned kNN neighbors: depthwise (CPE) or full."""

    def __init__(self, in_channels, out_channels=None, depthwise=False):
        super().__init__()
        self.depthwise = depthwise
        out_channels = in_channels if depthwise else out_channels
        shape = (27, in_channels) if depthwise else (27, in_channels, out_channels)
        self.weight = nn.Parameter(torch.empty(shape))
        nn.init.trunc_normal_(self.weight, std=(27 * (1 if depthwise else in_channels)) ** -0.5)

    def forward(self, x, level):
        neighbors = gather_points(x, level.neighbors) * level.share.unsqueeze(-1)  # B, N, k, C
        binned = torch.zeros(*x.shape[:2], 27, x.shape[-1], device=x.device, dtype=x.dtype)
        binned = binned.scatter_add(2, level.cells.unsqueeze(-1).expand_as(neighbors), neighbors)
        if self.depthwise:
            return (binned * self.weight).sum(2)
        return torch.einsum("bnoc,ocd->bnd", binned, self.weight)


def batch_norm(norm, x):
    return norm(x.transpose(1, 2)).transpose(1, 2)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        self.conv = VoxelConv(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x, level):
        return self.activation(batch_norm(self.norm, self.conv(x, level)))


class RelativePositionBias(nn.Module):
    """OctFormer's RPE: a per-axis table over clamped voxel offsets within a window."""

    def __init__(self, patch_size, heads, dilation):
        super().__init__()
        self.bound = int(0.8 * patch_size * dilation ** 0.5)
        self.table = nn.Parameter(torch.zeros(3 * (2 * self.bound + 1), heads))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, offsets):  # W x K x K x 3 integer offsets -> W x H x K x K
        axis = torch.arange(3, device=offsets.device) * (2 * self.bound + 1)
        index = offsets.clamp(-self.bound, self.bound) + self.bound + axis
        bias = self.table.index_select(0, index.reshape(-1)).view(*index.shape, -1)
        return bias.sum(3).permute(0, 3, 1, 2)


class WindowAttention(nn.Module):
    """Self-attention within windows along the curve, with OctFormer's RPE.

    ``patch_size=0`` attends over all points without RPE: positions then come
    from the blocks' CPE (as in Point Transformer V3), and the unmasked
    attention runs on PyTorch's fused kernels.
    """

    def __init__(self, dim, heads, patch_size, dilation):
        super().__init__()
        self.heads, self.patch_size, self.dilation = heads, patch_size, dilation
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rpe = RelativePositionBias(patch_size, heads, dilation) if patch_size else None

    def windows(self, length):
        """Window size and dilation at a level of ``length`` points."""
        if not self.patch_size:
            return length, 1
        size = min(self.patch_size, length)
        return size, max(1, min(self.dilation, length // size))

    def forward(self, x, level):
        batch, length, channels = x.shape
        if not self.patch_size:
            q, k, v = self.qkv(x).view(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.proj(out.transpose(1, 2).reshape(batch, length, channels))
        size, dilation = self.windows(length)
        block = size * dilation
        padding = (-length) % block
        valid = x.new_ones(batch, length, dtype=torch.bool)
        grid = level.grid
        if padding:
            x, grid = F.pad(x, (0, 0, 0, padding)), F.pad(grid, (0, 0, 0, padding))
            valid = F.pad(valid, (0, padding))

        def partition(t):  # B x L x C -> windows x K x C; window d of a block takes points d, d + D, ...
            return t.reshape(batch, -1, size, dilation, t.shape[-1]).transpose(2, 3).reshape(-1, size, t.shape[-1])

        tokens, grid, valid = partition(x), partition(grid), partition(valid.unsqueeze(-1))[..., 0]
        q, k, v = self.qkv(tokens).view(len(tokens), size, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        bias = self.rpe(grid.unsqueeze(2) - grid.unsqueeze(1))
        bias = bias.masked_fill(~valid[:, None, None, :], MASKED)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        out = out.transpose(1, 2).reshape(batch, -1, dilation, size, channels).transpose(2, 3)
        return self.proj(out.reshape(batch, -1, channels)[:, :length])


class DropPath(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        keep = x.new_empty(x.shape[0], 1, 1).bernoulli_(1 - self.rate)
        return x * keep / (1 - self.rate)


class OctFormerBlock(nn.Module):
    """x + CPE(x), then pre-norm window attention and MLP, as in OctFormer."""

    def __init__(self, dim, heads, patch_size, dilation, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.cpe = VoxelConv(dim, depthwise=True)
        self.cpe_norm = nn.BatchNorm1d(dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, patch_size, dilation)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, level):
        x = x + batch_norm(self.cpe_norm, self.cpe(x, level))
        x = x + self.drop_path(self.attention(self.norm1(x), level))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Downsample(nn.Module):
    """Pool ``stride`` consecutive points along the curve: a linear map of each
    child's features and offset from the parent, max-pooled, then normalized."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.linear = nn.Linear(in_channels + 3, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, level):
        batch, length, _ = x.shape
        padding = (-length) % self.stride
        xyz = level.xyz
        if padding:  # Repeat the last point so every parent has `stride` children.
            x = torch.cat((x, x[:, -1:].expand(-1, padding, -1)), 1)
            xyz = torch.cat((xyz, xyz[:, -1:].expand(-1, padding, -1)), 1)
        xyz = xyz.view(batch, -1, self.stride, 3)
        parents = xyz.mean(2)
        offsets = (xyz - parents.unsqueeze(2)) / level.voxel
        children = torch.cat((x.view(batch, -1, self.stride, x.shape[-1]), offsets), -1)
        return batch_norm(self.norm, self.linear(children).amax(2)), parents


class OctFormer(nn.Module):
    """Batched OctFormer encoder with an FPN head to per-point features.

    Defaults are a small configuration for 1-2k-point objects; the paper uses
    channels (96, 192, 384, 384), blocks (2, 2, 18, 2) and heads
    (6, 12, 24, 24) for ScanNet, with windows at every stage. Here stages
    from ``full_attention_from`` on use full attention (set it to
    ``len(blocks)`` for windows everywhere). ``voxel`` is the level-0 voxel
    size for clouds normalized to the unit sphere.
    """

    def __init__(self, emb_dims=1024, channels=(64, 128, 256, 256), blocks=(2, 2, 6, 2), heads=(4, 8, 16, 16),
                 patch_size=32, dilation=4, stride=4, k=16, voxel=0.0625, curve="z", drop_path=0.1,
                 fpn_channels=128, mlp_ratio=4.0, full_attention_from=1):
        super().__init__()
        if not len(channels) == len(blocks) == len(heads) or curve not in CURVES or stride < 2 or k < 1:
            raise ValueError("Invalid OctFormer configuration")
        if any(c % h for c, h in zip(channels, heads)) or voxel <= 0:
            raise ValueError("Channels must divide into heads, and the voxel size must be positive")
        self.curve, self.k, self.voxel, self.emb_dims = curve, k, voxel, emb_dims
        self.stem = nn.ModuleList([ConvBnAct(3, channels[0]), ConvBnAct(channels[0], channels[0])])
        rates = torch.linspace(0, drop_path, sum(blocks)).tolist()
        self.stages = nn.ModuleList()
        for stage, (dim, count, head) in enumerate(zip(channels, blocks, heads)):
            first = sum(blocks[:stage])
            # Windows (alternately dilated) where points are many, full attention once they are few.
            full = stage >= full_attention_from
            self.stages.append(nn.ModuleList([
                OctFormerBlock(dim, head, 0 if full else patch_size, 1 if full or i % 2 == 0 else dilation,
                               mlp_ratio, rates[first + i])
                for i in range(count)]))
        self.downsamples = nn.ModuleList([Downsample(channels[i], channels[i + 1], stride)
                                          for i in range(len(channels) - 1)])
        self.lateral = nn.ModuleList([nn.Linear(dim, fpn_channels) for dim in channels])
        self.fpn = nn.ModuleList([ConvBnAct(fpn_channels, fpn_channels) for _ in channels])
        self.out = nn.Linear(fpn_channels, emb_dims)
        self.out_norm = nn.BatchNorm1d(emb_dims)
        self.level_dims = (fpn_channels,) * len(channels)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, return_levels=False):
        """``x``: B x 3 x N. Returns (B x E global, B x E x N per-point[, FPN levels B x F x N])."""
        xyz = x.transpose(1, 2).contiguous()
        levels, features, parents = [Level(xyz, self.voxel, self.curve, self.k)], [], []
        h = levels[0].xyz - levels[0].xyz.mean(1, keepdim=True)
        for conv in self.stem:
            h = conv(h, levels[0])
        for stage, blocks in enumerate(self.stages):
            level = levels[-1]
            for block in blocks:
                h = block(h, level)
            features.append(h)
            if stage < len(self.downsamples):
                h, pooled = self.downsamples[stage](h, level)
                levels.append(Level(pooled, self.voxel * 2 ** (stage + 1), self.curve, self.k))
                # Children map to their parent's position in the next level's curve order.
                position = levels[-1].order.argsort(dim=1)
                stride = self.downsamples[stage].stride
                parents.append(position.gather(1, torch.arange(level.xyz.shape[1], device=x.device)
                                               .div(stride, rounding_mode="floor").expand(len(x), -1)))
                h = gather_points(h, levels[-1].order)

        def to_full(t, stage):  # Unpool level `stage` features to level 0 (curve order).
            for index in reversed(parents[:stage]):
                t = gather_points(t, index)
            return t

        top = self.lateral[-1](features[-1])
        out = [to_full(self.fpn[-1](top, levels[-1]), len(features) - 1)]
        for stage in range(len(features) - 2, -1, -1):
            top = gather_points(top, parents[stage]) + self.lateral[stage](features[stage])
            out.append(to_full(self.fpn[stage](top, levels[stage]), stage))
        inverse = levels[0].order.argsort(dim=1)  # Back to input order.
        out = [gather_points(t, inverse) for t in out]
        per_point = F.gelu(batch_norm(self.out_norm, self.out(sum(out)))).transpose(1, 2)
        global_features = per_point.amax(-1)
        if return_levels:
            return global_features, per_point, tuple(t.transpose(1, 2) for t in reversed(out))
        return global_features, per_point

