"""Precompute DINOv2 patch features of the paired multi-view images.

Each view's centered square crop (``--crop`` pixels; objects have unit radius,
so the crop holds every object pixel) is resized to ``--input-size`` and
encoded by DINOv2. Its grid of patch features is L2-normalized and projected
onto a PCA basis fitted on foreground patches of a subset of objects.

    --stage fit      writes OUTPUT/pca.pt
    --stage extract  writes OUTPUT/<relative path>.npz with ``features``
                     (V x C x G x G float16, zero where no object pixel falls)
                     and ``coverage`` (V x G x G float16, foreground fraction
                     of each patch, so thin parts keep their features)

The PCA is fitted on patches with at least ``--fit-coverage`` foreground.

``feature_intrinsics`` maps the archive intrinsics to the G x G feature grid.
"""

import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.multiview import crop_offset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def foreground(archive):
    """V x H x W foreground of the views: rendered depth, or non-black pixels for archives without depth."""
    if "depths" in archive.files:
        return np.asarray(archive["depths"], dtype=np.float32) > 0
    return np.asarray(archive["images"]).max(-1) > 0


class Views(Dataset):
    def __init__(self, root, entries):
        self.root, self.entries = Path(root), entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        with np.load(self.root / self.entries[index]) as archive:
            images = torch.from_numpy(np.asarray(archive["images"]))
            mask = torch.from_numpy(foreground(archive))
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[1] != images.shape[2]:
            raise ValueError(f"{self.entries[index]}: expected V x S x S x 3 images")
        return {"images": images, "mask": mask, "index": index}


def preprocess(images, mask, crop, input_size, grid):
    """Crop, resize and normalize B x S x S x 3 uint8 views; return them and each patch's foreground fraction."""
    offset = crop_offset(images.shape[1], crop)
    window = slice(offset, offset + crop)
    x = images[:, window, window].permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(input_size, input_size), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    coverage = F.adaptive_avg_pool2d(mask[:, window, window].unsqueeze(1).float(), grid)[:, 0]
    return (x - mean) / std, coverage


def load_dinov2(name, hub_dir, device):
    model = torch.hub.load(str(Path(hub_dir).expanduser()), name, source="local", pretrained=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode(model, x, grid):
    """B x 3 x I x I normalized images -> B x G x G x D L2-normalized patch features."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        tokens = model.forward_features(x)["x_norm_patchtokens"]
    if tokens.shape[1] != grid * grid:
        raise ValueError(f"Expected {grid * grid} patch tokens, got {tokens.shape[1]}")
    return F.normalize(tokens.float(), dim=-1).view(len(x), grid, grid, -1)


def batches(root, entries, args):
    loader = DataLoader(Views(root, entries), batch_size=args.objects_per_batch, num_workers=args.workers,
                        collate_fn=lambda items: items)
    for items in loader:
        images = torch.cat([item["images"] for item in items]).to(args.device)
        mask = torch.cat([item["mask"] for item in items]).to(args.device)
        x, coverage = preprocess(images, mask, args.crop, args.input_size, args.grid)
        yield items, encode(args.model_instance, x, args.grid), coverage


def fit(entries, args):
    """PCA of foreground patch features of a seeded subset of objects."""
    random.Random(args.seed).shuffle(entries)
    total = moment = count = None
    for _, features, coverage in batches(args.root, entries[:args.fit_objects], args):
        selected = features[coverage >= args.fit_coverage].double()
        if total is None:
            dim = selected.shape[-1]
            total, moment, count = selected.new_zeros(dim), selected.new_zeros(dim, dim), 0
        total += selected.sum(0)
        moment += selected.T @ selected
        count += len(selected)
    if not count:
        raise ValueError("No foreground patches in the fitting subset")
    mean = total / count
    values, vectors = torch.linalg.eigh(moment / count - torch.outer(mean, mean))
    order = values.argsort(descending=True)[:args.components]
    state = {"mean": mean.float().cpu(), "components": vectors[:, order].float().cpu(),
             "explained": float(values[order].sum() / values.sum()), "patches": count,
             "objects": min(len(entries), args.fit_objects), "model": args.model, "input_size": args.input_size,
             "crop": args.crop, "grid": args.grid}
    torch.save(state, args.output / "pca.pt")
    print(f"PCA on {count} foreground patches: {args.components} components keep "
          f"{100 * state['explained']:.1f}% of the variance", flush=True)


def extract(entries, args):
    state = torch.load(args.output / "pca.pt", map_location=args.device, weights_only=True)
    for key in ("model", "input_size", "crop", "grid"):
        if state[key] != getattr(args, key):
            raise ValueError(f"{key} differs from the fitted PCA: {state[key]} vs {getattr(args, key)}")
    mean, components = state["mean"], state["components"]
    shard, shards = args.shard
    entries = [entry for entry in entries[shard::shards]
               if not (args.output / entry).exists()]
    print(f"Shard {shard}/{shards}: {len(entries)} archives to extract", flush=True)
    for done, (items, features, coverage) in enumerate(batches(args.root, entries, args)):
        projected = ((features - mean) @ components) * (coverage > 0).unsqueeze(-1)
        projected = projected.permute(0, 3, 1, 2).half().cpu().numpy()
        coverage, start = coverage.half().cpu().numpy(), 0
        for item in items:
            views = len(item["images"])
            destination = args.output / entries[item["index"]]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, features=projected[start:start + views],
                                    coverage=coverage[start:start + views])
            os.replace(temporary, destination)
            start += views
        if done % 100 == 0:
            print(f"{(done + 1) * args.objects_per_batch} / {len(entries)}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("fit", "extract"), required=True)
    p.add_argument("--root", type=Path, required=True, help="Paired archives from prepare_shapenet.py")
    p.add_argument("--split", type=Path, help="Archive paths relative to --root (default: all archives)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="dinov2_vitb14")
    p.add_argument("--hub-dir", default="~/.cache/torch/hub/facebookresearch_dinov2_main")
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--crop", type=int, default=182)
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--fit-objects", type=int, default=1000)
    p.add_argument("--fit-coverage", type=float, default=0.5, help="Least foreground fraction of fitted patches")
    p.add_argument("--objects-per-batch", type=int, default=8)
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def run(args):
    if args.input_size % 14:
        raise ValueError("DINOv2 needs an input size divisible by its 14-pixel patches")
    if not 0 <= args.shard[0] < args.shard[1]:
        raise ValueError("Shard index must lie in [0, COUNT)")
    args.grid = args.input_size // 14
    args.output.mkdir(parents=True, exist_ok=True)
    if args.split is not None:
        entries = [line.strip() for line in args.split.read_text().splitlines() if line.strip()]
    else:
        entries = sorted(str(path.relative_to(args.root)) for path in args.root.rglob("*.npz"))
    if not entries:
        raise ValueError("No archives to process")
    if not hasattr(args, "model_instance"):
        args.model_instance = load_dinov2(args.model, args.hub_dir, args.device)
    (fit if args.stage == "fit" else extract)(entries, args)


if __name__ == "__main__":
    run(parser().parse_args())
