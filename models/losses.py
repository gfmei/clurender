"""CluRender objectives, following IJCV 2024 equations (2), (11), (14), (18)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.common import sinkhorn, square_distance


def round_transport_plan(plan, row_marginal, column_marginal):
    """Project a nonnegative approximate plan onto its prescribed marginals.

    Algorithm 2 of Altschuler et al. (2017), https://arxiv.org/abs/1705.09634:
    cap excess row/column mass, then fill the remaining deficits with a
    rank-one correction. This enforces probability targets even when a short
    Sinkhorn run has not converged; it does not guarantee OT optimality.
    """
    tiny = torch.finfo(plan.dtype).tiny
    row_scale = (row_marginal / plan.sum(-1).clamp_min(tiny)).clamp_max(1)
    plan = plan * row_scale.unsqueeze(-1)
    column_scale = (column_marginal / plan.sum(-2).clamp_min(tiny)).clamp_max(1)
    plan = plan * column_scale.unsqueeze(-2)
    row_deficit = (row_marginal - plan.sum(-1)).clamp_min(0)
    column_deficit = (column_marginal - plan.sum(-2)).clamp_min(0)
    missing = row_deficit.sum(-1, keepdim=True).clamp_min(tiny)
    return plan + (row_deficit / missing).unsqueeze(-1) * column_deficit.unsqueeze(-2)


class BalancedClustering(nn.Module):
    """Learn geometric partitions using detached, balanced soft pseudo-labels.

    Sinkhorn stops once the relative point-marginal error is below
    ``tolerance`` (zero runs all ``iterations``). With a small ``epsilon``,
    separated prototypes can need hundreds of iterations to converge.
    """

    def __init__(self, dim, num_clusters=64, epsilon=0.001, iterations=2000,
                 orthogonal_weight=0.01, round_assignments=True, tolerance=0.01):
        super().__init__()
        if (dim < 2 or num_clusters < 1 or not math.isfinite(epsilon) or epsilon <= 0
                or iterations < 1 or not math.isfinite(orthogonal_weight) or orthogonal_weight < 0
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Invalid clustering dimensions or Sinkhorn settings")
        hidden = dim // 2
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, num_clusters),
        )
        self.epsilon = epsilon
        self.iterations = iterations
        self.orthogonal_weight = orthogonal_weight
        self.round_assignments = round_assignments
        self.tolerance = tolerance

    def forward(self, features, points, return_details=False):
        logits = self.head(features.transpose(1, 2))  # B, N, J
        scores = logits.softmax(-1)
        xyz = points.transpose(1, 2)
        prototypes = scores.transpose(1, 2) @ xyz
        prototypes = prototypes / scores.sum(1).unsqueeze(-1).clamp_min(1e-8)
        batch, count, clusters = scores.shape
        with torch.no_grad():
            cost = square_distance(xyz, prototypes.detach()).clamp_min(0)
            p = cost.new_full((batch, count), 1.0 / count)
            q = cost.new_full((batch, clusters), 1.0 / clusters)
            plan, _ = sinkhorn(cost, p, q, self.epsilon, self.tolerance, self.iterations, check_every=10,
                               anneal=True)
            marginal_error = (count * plan.sum(-1) - 1).abs().amax()
            if self.round_assignments:
                plan = round_transport_plan(plan, p, q)
            labels = count * plan
        cross_entropy = -(labels * logits.log_softmax(-1)).sum(-1).mean()
        # Directions from each cloud's centroid, not the world origin, keep the
        # regularizer invariant to where the object sits in world coordinates.
        normalized = F.normalize(prototypes - xyz.mean(1, keepdim=True), dim=-1)
        gram = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(clusters, device=gram.device, dtype=gram.dtype)
        orthogonal = torch.linalg.matrix_norm(gram - identity, ord="fro").mean()
        loss = cross_entropy + self.orthogonal_weight * orthogonal
        if return_details:
            return loss, {"labels": labels, "prototypes": prototypes,
                          "cross_entropy": cross_entropy, "orthogonal": orthogonal,
                          "unrounded_marginal_error": marginal_error}
        return loss


def image_samples(images, position_weight=0.5):
    """Embed BCHW RGB images as uniformly weighted (RGB, x, y) samples.

    Coordinates are pixel centers normalized to [0, 1]. Squared Euclidean
    distance equals (1-lambda) * color distance + lambda * position distance.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images with shape (B, 3, H, W)")
    if not 0 <= position_weight <= 1:
        raise ValueError("position_weight must lie in [0, 1]")
    batch, _, height, width = images.shape
    y, x = torch.meshgrid(
        (torch.arange(height, device=images.device, dtype=images.dtype) + 0.5) / height,
        (torch.arange(width, device=images.device, dtype=images.dtype) + 0.5) / width,
        indexing="ij",
    )
    coords = torch.stack((x, y), -1).reshape(1, -1, 2).expand(batch, -1, -1)
    colors = images.flatten(2).transpose(1, 2)
    return torch.cat((colors * math.sqrt(1 - position_weight),
                      coords * math.sqrt(position_weight)), dim=-1)


class ImageWassersteinLoss(nn.Module):
    """Entropy-regularized approximation of the paper's full image OT loss.

    GeomLoss's online backend avoids materializing a (H*W)^2 cost matrix.
    No pixels are subsampled. This is regularized OT, not sliced Wasserstein.
    """

    def __init__(self, position_weight=0.5, blur=0.01, backend="auto"):
        super().__init__()
        if not 0 <= position_weight <= 1 or blur <= 0:
            raise ValueError("Invalid image transport settings")
        if backend not in ("auto", "tensorized", "online"):
            raise ValueError("OT backend must be auto, tensorized, or online")
        try:
            from geomloss import SamplesLoss
        except ImportError as exc:
            raise ImportError("Sinkhorn fitting requires geomloss; install requirements-pretrain.txt") from exc
        self.position_weight = position_weight
        self.backend = backend
        # A fixed bound also handles identical 1-pixel images, whose inferred
        # diameter would be zero and would make GeomLoss's schedule undefined.
        diameter = math.sqrt(3 * (1 - position_weight) + 2 * position_weight)
        self.tensorized = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False,
                                      diameter=diameter, backend="tensorized")
        self.online = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False,
                                  diameter=diameter, backend="online")

    def forward(self, rendered, target):
        if rendered.shape != target.shape:
            raise ValueError("Rendered and target image shapes must match")
        x = image_samples(rendered, self.position_weight)
        y = image_samples(target, self.position_weight)
        online = self.backend == "online" or (self.backend == "auto" and x.shape[1] > 1024)
        if online:
            try:
                import pykeops  # noqa: F401
            except ImportError as exc:
                raise ImportError("Full-resolution image OT requires pykeops; use small images for CPU checks") from exc
        # GeomLoss p=2 uses half the squared distance. Restore the paper's scale.
        criterion = self.online if online else self.tensorized
        return 2 * criterion(x.contiguous(), y.contiguous()).mean()


class _SlicedSquaredW2(torch.autograd.Function):
    """Sliced squared W2 between equally sized, uniformly weighted sample sets.

    Each projection sorts N scalars, so no N x N cost matrix is ever formed.
    The gradient with respect to ``x`` is accumulated during the forward pass;
    only that (B, N, D) buffer is kept, not the per-projection sort indices.
    ``y`` is a target and receives no gradient. With ``key_dtype`` (e.g.
    float16), the order comes from lower-precision keys, which halves the
    radix-sort passes; values and gradients stay in full precision, and only
    nearly equal values may swap places.
    """

    @staticmethod
    def forward(ctx, x, y, directions, chunk_size, key_dtype=None):
        batch, count, _ = x.shape
        total = x.new_zeros(())
        grad = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        x_t, y_t = x.transpose(1, 2), y.transpose(1, 2)
        for projection in directions.split(chunk_size):
            # (B, projections, N): sorting along the last, contiguous dimension is fastest.
            projected, target = projection @ x_t, projection @ y_t
            order = (projected if key_dtype is None else projected.to(key_dtype)).argsort(dim=-1)
            target_order = (target if key_dtype is None else target.to(key_dtype)).argsort(dim=-1)
            difference = projected.gather(-1, order) - target.gather(-1, target_order)
            total += difference.square().mean(dim=(0, 2)).sum()
            if grad is not None:
                # Return each sorted difference to its sample, then back to sample space.
                grad += torch.empty_like(difference).scatter_(-1, order, difference).transpose(1, 2) @ projection
        scale = 1 / len(directions)
        if grad is not None:
            grad *= 2 * scale / (batch * count)
        ctx.save_for_backward(grad)
        return total * scale

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad_output * grad, None, None, None, None


class SlicedWassersteinDistance(nn.Module):
    """Sliced squared W2 between images as (RGB, position) samples.

    Random unit directions project the 5-D samples; memory is linear in the
    number of pixels. Kept channels-last for the original public helper.
    """

    def __init__(self, num_projections=128, position_weight=0.5, chunk_size=128, key_dtype=torch.float16):
        super().__init__()
        if num_projections < 1 or chunk_size < 1:
            raise ValueError("Projection counts must be positive")
        self.num_projections = num_projections
        self.position_weight = position_weight
        self.chunk_size = chunk_size
        self.key_dtype = key_dtype

    def forward(self, x, y):
        if x.shape != y.shape or x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("Expected equal BHWC RGB images")
        x = image_samples(x.permute(0, 3, 1, 2), self.position_weight)
        y = image_samples(y.permute(0, 3, 1, 2), self.position_weight)
        directions = F.normalize(torch.randn(self.num_projections, 5, device=x.device, dtype=x.dtype), dim=-1)
        # Squared W2 has a finite zero gradient for identical images.
        return _SlicedSquaredW2.apply(x, y.detach(), directions, self.chunk_size, self.key_dtype)


class SlicedImageLoss(SlicedWassersteinDistance):
    def forward(self, rendered, target):
        return super().forward(rendered.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1))
