"""ShapeNetPart linear probing of a strictly loaded, frozen CluRender encoder.

Train a 50-way pointwise linear head on train+val; test only after the fixed
training budget. Cached float16 point features avoid repeated encoder passes.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.shapenetpart import CATEGORIES, PartSegMetrics, ShapeNetPartText
from models.dgcnn import DGCNN
from models.pointnet import PointNet


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_encoder(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "args" not in state or "model" not in state:
        raise ValueError("--pretrained requires a full main_pretrain.py checkpoint, such as last.pth")
    config = state["args"]
    if config["model"] == "dgcnn":
        encoder = DGCNN(config["emb_dims"], config["k"], num_cls=-1)
    elif config["model"] == "pointnet":
        encoder = PointNet(config["emb_dims"], feature_transform=True, feat_type="global")
    else:
        raise ValueError("Unsupported encoder")
    weights = {name.removeprefix("backbone."): value for name, value in state["model"].items()
               if name.startswith("backbone.")}
    encoder.load_state_dict(weights, strict=True)
    encoder.eval().requires_grad_(False)
    return encoder, config, weights


def category_mask(device="cpu"):
    mask = torch.zeros(16, 50, dtype=torch.bool, device=device)
    for i, (_, _, start, count) in enumerate(CATEGORIES):
        mask[i, start:start + count] = True
    return mask


def predict_parts(logits, categories, mask):
    return logits.masked_fill(~mask[categories, None, :], -torch.inf).argmax(-1)


def cache_features(encoder, dataset, cache_root, encoder_hash, dim, args):
    signature = {"format": 1, "encoder_sha256": encoder_hash, "dataset": dataset.fingerprint,
                 "dim": dim, "dtype": "float16"}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache_root / key
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = directory / "complete.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["signature"] != signature:
                raise ValueError("Feature cache metadata mismatch")
            return directory
        shape = (len(dataset), args.num_points, dim)
        features = np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype=np.float16, shape=shape)
        labels = np.lib.format.open_memmap(directory / "parts.npy", mode="w+", dtype=np.uint8, shape=shape[:2])
        points = np.lib.format.open_memmap(directory / "points.npy", mode="w+", dtype=np.float32,
                                         shape=(len(dataset), args.num_points, 3))
        np.save(directory / "categories.npy", dataset.categories)
        loader = DataLoader(dataset, batch_size=args.extract_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=args.device.startswith("cuda"))
        first_error, offset = None, 0
        start = time.perf_counter()
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                xyz = batch["points"].to(args.device, non_blocking=True)
                full = encoder(xyz)[1].transpose(1, 2).contiguous()
                half = full.half()
                if not torch.isfinite(half).all():
                    raise FloatingPointError("Encoder features overflowed float16 cache")
                if first_error is None:
                    first_error = (half.float() - full).abs().max().item()
                end = offset + len(xyz)
                features[offset:end] = half.cpu().numpy()
                labels[offset:end] = batch["parts"].numpy()
                points[offset:end] = batch["points"].transpose(1, 2).numpy()
                offset = end
                if step % 25 == 0 or offset == len(dataset):
                    print(f"Features {offset}/{len(dataset)} objects, {time.perf_counter() - start:.1f}s", flush=True)
        features.flush()
        labels.flush()
        points.flush()
        manifest.write_text(json.dumps({"signature": signature, "ids": dataset.ids,
                                       "shape": shape, "first_batch_fp16_max_error": first_error}, indent=2) + "\n")
    return directory


def load_cache(directory, device, chunk=256):
    # Keep CPU data in RAM, or the complete feature matrix on the allocated GPU
    # when it fits. Training one linear layer needs little additional memory.
    # GPU copies stream from a memory map, so host memory stays far below the
    # size of the feature matrix.
    mapped = np.load(directory / "features.npy", mmap_mode="r")
    parts = torch.from_numpy(np.load(directory / "parts.npy").astype(np.int64))
    categories = torch.from_numpy(np.load(directory / "categories.npy"))
    residence = torch.device("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        required = mapped.size * mapped.itemsize + parts.numel() * 8
        if required + 3 * 1024**3 < free:
            residence = torch.device(device)
    if residence.type == "cuda":
        features = torch.empty(mapped.shape, dtype=torch.float16, device=residence)
        for start in range(0, len(mapped), chunk):
            features[start:start + chunk] = torch.from_numpy(np.array(mapped[start:start + chunk]))
        parts, categories = parts.to(residence), categories.to(residence)
    else:
        features = torch.from_numpy(np.load(directory / "features.npy"))
    del mapped
    print(f"Loaded {len(features)} objects; feature storage: {residence}", flush=True)
    return features, parts, categories


def evaluate(head, cache, batch_size, device):
    features, parts, categories = cache
    metrics = PartSegMetrics()
    mask = category_mask(device)
    head.eval()
    predictions = np.empty(parts.shape, dtype=np.uint8)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            logits = head(features[start:stop].to(device).float())
            pred = predict_parts(logits, categories[start:stop].to(device), mask).cpu().numpy()
            metrics.update(pred, parts[start:stop].cpu().numpy(), categories[start:stop].cpu().numpy())
            predictions[start:stop] = pred
    return metrics.compute(), predictions


def run(args):
    if min(args.num_points, args.epochs, args.batch_size, args.extract_batch_size, args.lr_step) < 1:
        raise ValueError("Point counts, epochs, batch sizes and scheduler intervals must be positive")
    if args.workers < 0 or args.seed < 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 < args.lr_gamma <= 1:
        raise ValueError("Invalid execution or optimizer settings")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requires a GPU allocation")
    output, cache_root = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.resume:
        raise ValueError("Output already contains a run; use --resume or choose a new --output")
    start = time.perf_counter()
    encoder, config, encoder_weights = load_encoder(args.pretrained)
    if config["model"] == "dgcnn" and args.num_points < config["k"]:
        raise ValueError("num-points is smaller than the encoder's k")
    encoder_hash = file_digest(args.pretrained)
    train_data = ShapeNetPartText(args.root, "trainval", args.num_points, args.seed, args.exclude_overlap)
    test_data = ShapeNetPartText(args.root, "test", args.num_points, args.seed, args.exclude_overlap)
    (output / "split_audit.json").write_text(json.dumps(train_data.split_audit, indent=2) + "\n")
    dim = config["emb_dims"]
    protocol = {"encoder_sha256": encoder_hash, "train_fingerprint": train_data.fingerprint,
                "test_fingerprint": test_data.fingerprint, "points": args.num_points,
                "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
                "weight_decay": args.weight_decay, "lr_step": args.lr_step, "lr_gamma": args.lr_gamma,
                "exclude_overlap": args.exclude_overlap}
    previous = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if previous is not None and previous["protocol"] != protocol:
        raise ValueError("Resume protocol or encoder differs from this run")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"Frozen CluRender/{config['model']}: {len(train_data)} train+val, {len(test_data)} test; "
          f"{args.num_points} points, {args.epochs} linear-head epochs", flush=True)
    encoder.to(args.device)
    train_cache = cache_features(encoder, train_data, cache_root, encoder_hash, dim, args)
    # Test labels/features never participate in head optimization or selection.
    test_cache = cache_features(encoder, test_data, cache_root, encoder_hash, dim, args)
    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    features, labels, _ = load_cache(train_cache, args.device)
    torch.manual_seed(args.seed)
    head = nn.Linear(dim, 50).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    if previous is not None:
        head.load_state_dict(previous["head"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        generator.set_state(previous["generator"])
        start_epoch = previous["epoch"]
    x = target = logits = loss = order = None
    for epoch in range(start_epoch, args.epochs):
        head.train()
        epoch_start = time.perf_counter()
        total_loss = torch.zeros((), device=args.device)
        order = torch.randperm(len(features), generator=generator).to(features.device)
        for indices in order.split(args.batch_size):
            x = features[indices].to(args.device).float()
            target = labels[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 50), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * target.numel()
        mean_loss = (total_loss / labels.numel()).item()
        if not np.isfinite(mean_loss) or not all(torch.isfinite(p).all() for p in head.parameters()):
            raise FloatingPointError(f"Non-finite linear-head training at epoch {epoch + 1}")
        metrics = {"epoch": epoch + 1, "train_loss": mean_loss,
                   "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - epoch_start}
        scheduler.step()
        atomic_save({"epoch": epoch + 1, "head": head.state_dict(), "encoder": encoder_weights,
                     "encoder_config": config, "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                     "protocol": protocol, "args": vars(args)}, output / "last.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    del features, labels, order, x, target, logits, loss
    if not (output / "last.pth").exists() and previous is not None:
        atomic_save(previous, output / "last.pth")
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    results, predictions = evaluate(head, load_cache(test_cache, args.device), args.batch_size, args.device)
    np.save(output / "test_predictions.npy", predictions)
    results.update({"protocol": "Frozen encoder; one 50-way linear head; final epoch; category-masked predictions",
                    "epochs": args.epochs, "train_objects": len(train_data), "num_points": args.num_points,
                    "pretrained_checkpoint": str(Path(args.pretrained).resolve()), "encoder_sha256": encoder_hash,
                    "pretraining_data": config.get("root"), "pretraining_epochs": config.get("epochs"),
                    "encoder": config["model"], "feature_precision": "float16 cache, float32 linear head",
                    "sampling": "fixed seeded uniform sampling of normalized XYZ; normals unused",
                    "split_audit": train_data.split_audit,
                    "test_cache": str(test_cache.resolve()), "train_cache": str(train_cache.resolve()),
                    "seconds": time.perf_counter() - start})
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("TEST " + json.dumps(results), flush=True)
    return results


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True)
    cli.add_argument("--pretrained", required=True)
    cli.add_argument("--cache", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--resume")
    cli.add_argument("--exclude-overlap", action="store_true",
                     help="Keep test IDs intact; exclude them from fitting and deduplicate train/val")
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--num-points", type=int, default=2048)
    cli.add_argument("--epochs", type=int, default=100)
    cli.add_argument("--batch-size", type=int, default=24)
    cli.add_argument("--extract-batch-size", type=int, default=12)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--lr", type=float, default=.001)
    cli.add_argument("--weight-decay", type=float, default=.01)
    cli.add_argument("--lr-step", type=int, default=20)
    cli.add_argument("--lr-gamma", type=float, default=.5)
    return cli


if __name__ == "__main__":
    run(parser().parse_args())
