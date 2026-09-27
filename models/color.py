"""Point Transformer U-Net color prediction (CluRender section 5.1)."""

import torch
from torch import nn

from models.common import farthest_point_sample, index_points, square_distance


class PointAttention(nn.Module):
    """Local vector attention with relative position encoding and a residual."""

    def __init__(self, dim, neighbors=16):
        super().__init__()
        self.neighbors = neighbors
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attention = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.output = nn.Linear(dim, dim)

    def forward(self, points, features):
        count = points.shape[1]
        normalized = self.norm(features)
        query, key, value = self.query(normalized), self.key(normalized), self.value(normalized)
        outputs = []
        # Chunk neighborhoods so the pairwise distance matrix is not N x N.
        for start in range(0, count, 256):
            xyz = points[:, start:start + 256]
            with torch.no_grad():
                indices = square_distance(xyz, points).topk(min(self.neighbors, count), largest=False).indices
            position = self.position(xyz.unsqueeze(2) - index_points(points, indices))
            delta = query[:, start:start + 256].unsqueeze(2) - index_points(key, indices) + position
            attention = self.attention(delta).softmax(dim=2)
            outputs.append((attention * (index_points(value, indices) + position)).sum(2))
        return features + self.output(torch.cat(outputs, 1))


class TransformerDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, num_points, is_center=True):
        super().__init__()
        self.num_points = num_points
        self.is_center = is_center
        self.conv = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, points, features, num_points=None):
        count = min(num_points if num_points is not None else self.num_points, points.shape[1])
        if count < 1:
            raise ValueError("Each U-Net stage must contain at least one point")
        if count == points.shape[1]:
            return points, self.conv(features)
        with torch.no_grad():
            indices = farthest_point_sample(points, count, is_center=self.is_center)
        return index_points(points, indices), self.conv(index_points(features, indices))


class TransformerUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, xyz1, xyz2, features1, features2):
        if xyz1 is xyz2:
            interpolated = features2
        else:
            distances, indices = square_distance(xyz1, xyz2).clamp_min(0).topk(
                min(3, xyz2.shape[1]), largest=False)
            weights = 1 / distances.clamp_min(1e-8)
            weights = weights / weights.sum(-1, keepdim=True)
            interpolated = (index_points(features2, indices) * weights.unsqueeze(-1)).sum(2)
        if features1 is not None:
            interpolated = torch.cat((features1, interpolated), -1)
        return self.conv(interpolated)


class UNetTransformer(nn.Module):
    """Three down/up stages with cardinalities N, N/4, N/16 by default.

    Input: XYZ (B, N, 3), features (B, N, C). Output: logits (B, 3, N).
    ``num_heads`` remains accepted for compatibility; local vector attention
    assigns a separate neighbor distribution to each feature channel.
    """

    def __init__(self, in_channels, out_channels=3, num_samples_list=None,
                 d_dims=(512, 256, 128), u_dims=(128, 64, 32), num_heads=4,
                 is_center=True, neighbors=16):
        super().__init__()
        if len(d_dims) != 3 or len(u_dims) != 3:
            raise ValueError("The color U-Net requires three encoder and decoder dimensions")
        if num_samples_list is not None and (len(num_samples_list) != 3 or min(num_samples_list) < 1):
            raise ValueError("num_samples_list must contain three positive counts")
        self.num_samples_list = num_samples_list
        self.conv = nn.Linear(in_channels, d_dims[0])
        self.downsampling = nn.ModuleList()
        self.encoder_layers = nn.ModuleList()
        previous = d_dims[0]
        skip_dims = []
        for dim in d_dims:
            skip_dims.append(previous)
            self.downsampling.append(TransformerDownSampling(previous, dim, 1, is_center))
            self.encoder_layers.append(PointAttention(dim, neighbors))
            previous = dim
        self.upsampling = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for skip_dim, dim in zip(reversed(skip_dims), u_dims):
            self.upsampling.append(TransformerUpSampling(previous + skip_dim, dim))
            self.decoder_layers.append(PointAttention(dim, neighbors))
            previous = dim
        self.final_conv = nn.Linear(u_dims[-1], out_channels)

    def forward(self, points, features):
        if points.shape[:2] != features.shape[:2] or points.shape[-1] != 3:
            raise ValueError("Expected pointwise XYZ and features in BNC layout")
        count = points.shape[1]
        counts = self.num_samples_list or [count, max(1, count // 4), max(1, count // 16)]
        features = self.conv(features)
        skips = []
        for down, attention, size in zip(self.downsampling, self.encoder_layers, counts):
            skips.append((points, features))
            points, features = down(points, features, size)
            features = attention(points, features)
        for up, attention, (target_points, skip) in zip(self.upsampling, self.decoder_layers, reversed(skips)):
            features = up(target_points, points, skip, features)
            points = target_points
            features = attention(points, features)
        return self.final_conv(features).transpose(1, 2)
