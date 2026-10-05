"""Downstream models that share one pretrained DGCNN encoder.

Classification and part segmentation add task heads to the same encoder that
main_pretrain.py trains (``--model dgcnn``), so every task starts from one set
of pretrained weights.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.common import farthest_point_sample, index_points
from models.dgcnn import DGCNN


def load_encoder(encoder, path):
    """Load pretrained DGCNN encoder weights; every weight must match exactly.

    Accepts ``backbone.pth`` or a full main_pretrain.py checkpoint.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in state and "args" in state:
        if state["args"].get("model") != "dgcnn":
            raise ValueError(f"{path} was pretrained with --model {state['args'].get('model')}, not dgcnn")
        state = {name.removeprefix("backbone."): value for name, value in state["model"].items()
                 if name.startswith("backbone.")}
    encoder.load_state_dict(state)
    return len(state)


def augment(points, generator=None):
    """Random anisotropic scale in [2/3, 3/2] and translation in [-0.2, 0.2].

    This is Point-MAE's PointcloudScaleAndTranslate, also DGCNN's augmentation.
    """
    batch = len(points)
    scale = torch.empty(batch, 3, 1).uniform_(2 / 3, 3 / 2, generator=generator)
    shift = torch.empty(batch, 3, 1).uniform_(-.2, .2, generator=generator)
    return points * scale.to(points) + shift.to(points)


def subsample(points, count, pool=None, generator=None):
    """Subsample (B, 3, P) points with farthest point sampling, as Point-MAE does.

    Without ``pool``, return a deterministic FPS of ``count`` points (testing).
    With ``pool``, take an FPS of ``pool`` points and keep a random ``count``
    of them, the same selection for the whole batch (training and voting).
    """
    total = points.shape[-1]
    if count > total:
        raise ValueError(f"Cannot sample {count} of {total} points")
    xyz = points.transpose(1, 2)
    if pool is None:
        if count == total:
            return points
        return index_points(xyz, farthest_point_sample(xyz, count, is_center=True)).transpose(1, 2)
    pool = min(max(pool, count), total)
    indices = farthest_point_sample(xyz, pool) if pool < total else torch.arange(
        total, device=points.device).expand(len(points), -1)
    keep = torch.randperm(pool, generator=generator)[:count].to(indices.device)
    return index_points(xyz, indices[:, keep]).transpose(1, 2)


def build_optimizer(model, args):
    """SGD (DGCNN's fine-tuning default) or AdamW, with cosine decay to min_lr.

    ``args.encoder_lr_scale`` (default 1) multiplies the encoder's learning
    rate; the heads keep ``args.lr``.
    """
    scale = getattr(args, "encoder_lr_scale", 1.0)
    params = model.parameters()
    if scale != 1:
        encoder = {id(p) for p in model.encoder.parameters()}
        params = [{"params": [p for p in model.parameters() if id(p) not in encoder]},
                  {"params": list(model.encoder.parameters()), "lr": args.lr * scale}]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)


def set_encoder_frozen(model, frozen):
    """Freeze the encoder for linear probing: no gradients and fixed BatchNorm
    statistics. Call after model.train()."""
    model.encoder.requires_grad_(not frozen)
    if frozen:
        model.encoder.eval()


def part_mask(categories, device=None):
    """(16, 50) mask of the ShapeNetPart parts that belong to each category."""
    mask = torch.zeros(len(categories), 50, dtype=torch.bool, device=device)
    for index, (_, _, start, count) in enumerate(categories):
        mask[index, start:start + count] = True
    return mask


class DGCNNClassifier(nn.Module):
    """Encoder, then max- and average-pooled features into an MLP."""

    def __init__(self, num_classes, emb_dims=1024, k=20, dropout=0.5):
        super().__init__()
        self.encoder = DGCNN(emb_dims, k, num_cls=-1)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dims, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(.2), nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(.2), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, points):
        features = self.encoder(points)[1]
        return self.head(torch.cat((features.amax(-1), features.mean(-1)), 1))


class DGCNNPartSegmenter(nn.Module):
    """Encoder, then per-point MLP over multi-level, global and category features.

    Each point gets the four EdgeConv features, the final per-point feature,
    the global max-pooled feature, and an embedding of the object category.
    """

    def __init__(self, num_parts=50, num_categories=16, emb_dims=1024, k=40, dropout=0.5):
        super().__init__()
        self.num_categories = num_categories
        self.encoder = DGCNN(emb_dims, k, num_cls=-1)
        self.label = nn.Sequential(nn.Conv1d(num_categories, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(.2))
        width = 64 + 64 + 128 + 256 + 2 * emb_dims + 64
        self.head = nn.Sequential(
            nn.Conv1d(width, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(.2), nn.Dropout(dropout),
            nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(.2), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(.2),
            nn.Conv1d(128, num_parts, 1))

    def forward(self, points, categories):
        global_features, features, levels = self.encoder(points, return_levels=True)
        label = self.label(F.one_hot(categories, self.num_categories).to(points).unsqueeze(-1))
        context = torch.cat((global_features.unsqueeze(-1), label), 1).expand(-1, -1, points.shape[-1])
        return self.head(torch.cat((*levels, features, context), 1))
