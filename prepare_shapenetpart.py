"""Render calibrated synthetic RGB targets from ShapeNetPart XYZ alone.

Fixed material and world-space lighting shade PCA surface normals. No part
labels, segmentation visualizations, mesh textures or learned model are used.
These point-splat targets are an alternative to the paper's CAD renderings.
"""

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from datasets.multiview import orbit_cameras
from datasets.shapenetpart import split_ids
from models.common import points_to_ndc, transform_points_tsfm
from models.renderer import PointRenderer, RenderConfig


def select_sources(root, scope, limit=None):
    splits = split_ids(root, allow_overlap=True)
    held_out = set(splits["test"])
    available = {f"{p.parent.parent.name}/{p.stem}": p for p in root.glob("*/points/*.pts")}
    selected = (set(splits["train"]) | set(splits["val"])) if scope == "trainval" else set(available)
    selected -= held_out
    missing = selected - available.keys()
    if missing:
        raise FileNotFoundError(f"Missing XYZ clouds: {sorted(missing)[:5]}")
    identities = sorted(selected)
    if limit is not None:
        # A short validation run still covers every available category.
        groups = defaultdict(deque)
        for name in identities:
            groups[name.split('/')[0]].append(name)
        identities = []
        while groups and len(identities) < limit:
            for category in list(groups):
                identities.append(groups[category].popleft())
                if not groups[category]:
                    del groups[category]
                if len(identities) == limit:
                    break
    if not identities:
        raise ValueError("No non-test point clouds selected")
    return [(name, available[name]) for name in identities], sorted(held_out)


def geometry_rgb(path, neighbors, minimum_points):
    from scipy.spatial import cKDTree

    source = path.read_bytes()
    points = np.loadtxt(io.BytesIO(source), dtype=np.float32, ndmin=2)
    if points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"Expected at least three finite XYZ points: {path}")
    selection = None
    if len(points) < minimum_points:
        # Preserve original geometry and repeat points before normalization so
        # training points and RGB targets continue to share the same frame.
        seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        extra = minimum_points - len(points)
        indices = rng.choice(len(points), extra, replace=extra > len(points))
        selection = np.concatenate((np.arange(len(points)), indices))
    points -= points.mean(0) if selection is None else points[selection].mean(0)
    scale = np.linalg.norm(points, axis=1).max()
    if scale <= 1e-8:
        raise ValueError(f"Degenerate cloud: {path}")
    points /= scale
    _, indices = cKDTree(points).query(points, k=min(neighbors, len(points)), workers=1)
    local = points[indices]
    local -= local.mean(1, keepdims=True)
    covariance = local.transpose(0, 2, 1) @ local
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    lights = np.array([[1., 1., 2.], [-2., 1., -.5]], dtype=np.float32)
    lights /= np.linalg.norm(lights, axis=1, keepdims=True)
    # Two-sided diffuse shading is invariant to the arbitrary PCA normal sign.
    diffuse = np.abs(normals @ lights.T)
    intensity = .25 + .55 * diffuse[:, 0] + .20 * diffuse[:, 1]
    colors = intensity[:, None] * np.array([.65, .75, .90], dtype=np.float32)
    if selection is not None:
        # Estimate normals on original samples so duplicates do not collapse
        # the PCA neighborhood on especially sparse surfaces.
        points, colors = points[selection], colors[selection]
    return points, colors, hashlib.sha256(source).hexdigest()


def render_views(points, colors, transforms, intrinsics, renderer, device, view_batch):
    xyz = torch.from_numpy(points).to(device)
    rgb = torch.from_numpy(colors).to(device)
    images, coverage = [], []
    with torch.inference_mode():
        for start in range(0, len(transforms), view_batch):
            tsfm = torch.from_numpy(transforms[start:start + view_batch]).to(device)
            K = torch.from_numpy(intrinsics[start:start + view_batch]).to(device)
            camera = transform_points_tsfm(xyz[None].expand(len(tsfm), -1, -1), tsfm)
            ndc = points_to_ndc(camera, K, renderer.image_size)
            result = renderer(ndc, rgb[None].expand(len(tsfm), -1, -1).contiguous(), return_raster=False)
            images.append((result["feats"].permute(0, 2, 3, 1).clamp(0, 1) * 255)
                          .round().to(torch.uint8).cpu().numpy())
            coverage.extend(result["valid_rays"].cpu().tolist())
    if min(coverage) <= 0 or max(coverage) >= 1:
        raise ValueError("A rendered view is empty or fills the entire frame")
    return np.concatenate(images), coverage


def save_preview(images, path):
    from PIL import Image

    chosen = np.linspace(0, len(images), 4, endpoint=False, dtype=int)
    Image.fromarray(np.concatenate(images[chosen], axis=1)).save(path)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_archive(destination, **arrays):
    temporary = destination.with_suffix(".npz.tmp")
    # Level 1 preserves every array value and reduces offline compression time.
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)
    os.replace(temporary, destination)


def prepare(args):
    if min(args.views, args.image_size, args.view_batch, args.workers, args.minimum_points) < 1:
        raise ValueError("View, image, worker and point counts must be positive")
    if args.neighbors < 3 or args.minimum_points < 3 or (args.limit is not None and args.limit < 1):
        raise ValueError("Normals require at least three points/neighbors; limit must be positive")
    if (not all(math.isfinite(x) for x in (args.distance, args.elevation, args.radius, args.sigma))
            or args.distance <= 1 or abs(args.elevation) >= 89):
        raise ValueError("Invalid orbit camera settings")
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    sources, held_out = select_sources(root, args.scope, args.limit)
    records = [(name, path.stat().st_size, path.stat().st_mtime_ns) for name, path in sources]
    settings = {name: getattr(args, name) for name in ("views", "image_size", "distance", "elevation",
                "radius", "sigma", "points_per_pixel", "neighbors", "minimum_points", "scope", "limit", "renderer")}
    protocol = {"format": 1, "source_root": str(root), "settings": settings,
                "geometry_shading": "PCA normals; two-sided fixed world lights; fixed blue-gray material",
                "target_kind": "synthetic geometry RGB, not textured CAD renderings",
                "source_fingerprint": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
                "held_out_sha256": hashlib.sha256(json.dumps(held_out).encode()).hexdigest()}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "preparation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = output / "render_config.json"
        if config.exists() and json.loads(config.read_text()) != protocol:
            raise ValueError("Output contains different data/render settings; choose another output directory")
        atomic_json(config, protocol)
        transforms, intrinsics = orbit_cameras(args.views, args.image_size, args.distance, args.elevation)
        devices = args.devices or [args.device]
        renderers = [PointRenderer(RenderConfig(render_size=args.image_size, backend=args.renderer,
                                               radius=args.radius, sigma=args.sigma,
                                               points_per_pixel=args.points_per_pixel)) for _ in devices]
        previews = output / "previews"
        previews.mkdir(exist_ok=True)
        def complete(item):
            name, _ = item
            destination = output / (name + ".npz")
            if destination.exists():
                try:
                    with np.load(destination, allow_pickle=False) as data:
                        return str(data["preparation_signature"]) == signature
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    pass
            return False

        pending, skipped = [], 0
        # Shared-storage latency makes serial checks of thousands of ZIPs slow.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for checked, (item, valid) in enumerate(zip(sources, pool.map(complete, sources)), 1):
                if valid:
                    skipped += 1
                else:
                    pending.append(item)
                if checked % 5000 == 0:
                    print(f"Checked {checked}/{len(sources)} archives; {skipped} reusable", flush=True)
        started, written, failures = time.monotonic(), 0, []
        min_coverage, max_coverage = 1., 0.
        print(f"Render {len(sources)} non-test objects, {args.views} views each; {skipped} already prepared", flush=True)
        def prepare_one(name, path, device_index):
            points, colors, source_hash = geometry_rgb(path, args.neighbors, args.minimum_points)
            images, coverage = render_views(points, colors, transforms, intrinsics, renderers[device_index],
                                            devices[device_index], args.view_batch)
            destination = output / (name + ".npz")
            destination.parent.mkdir(exist_ok=True)
            save_archive(destination, points=points, images=images, tsfms=transforms, K=intrinsics,
                         source_sha256=source_hash, preparation_signature=signature)
            return images, coverage

        # Bounded parallelism overlaps geometry, GPU rasterization and compression.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator, queue = iter(enumerate(pending)), deque()

            def enqueue():
                item = next(iterator, None)
                if item is not None:
                    index, (name, path) = item
                    queue.append((name, pool.submit(prepare_one, name, path, index % len(devices))))

            for _ in range(2 * args.workers):
                enqueue()
            while queue:
                name, future = queue.popleft()
                try:
                    images, coverage = future.result()
                    preview = previews / (name.split('/')[0] + ".png")
                    if not preview.exists():
                        save_preview(images, preview)
                    min_coverage, max_coverage = min(min_coverage, min(coverage)), max(max_coverage, max(coverage))
                    written += 1
                except (ValueError, OSError, RuntimeError) as exc:
                    failures.append({"object": name, "error": str(exc)})
                    print(f"FAILED {name}: {exc}", flush=True)
                enqueue()
                if (written + len(failures)) % 100 == 0 or not queue:
                    print(f"Prepared {written + skipped}/{len(sources)}, failed {len(failures)}; "
                          f"{time.monotonic() - started:.1f}s", flush=True)
        report = {"protocol": protocol, "objects": len(sources), "views_per_object": args.views,
                  "total_rgb_views": len(sources) * args.views, "written": written, "reused": skipped,
                  "failures": failures, "excluded_test_objects": len(held_out),
                  "part_labels_used": False, "seconds": time.monotonic() - started,
                  "small_cloud_handling": "Deterministically repeat source points to minimum_points before normalization",
                  "new_view_coverage_range": [min_coverage, max_coverage] if written else None}
        atomic_json(output / "preparation.json", report)
        if failures:
            raise RuntimeError(f"{len(failures)} clouds failed; inspect preparation.json before training")
        split = output / "train.txt"
        temporary = split.with_suffix(".txt.tmp")
        temporary.write_text(''.join(name + ".npz\n" for name, _ in sources))
        os.replace(temporary, split)
        print(f"Complete: {len(sources) * args.views} RGB views; training split {split}", flush=True)
        return report


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, help="Extracted PartAnnotation directory with split lists")
    cli.add_argument("--output", required=True)
    cli.add_argument("--scope", choices=("all-nontest", "trainval"), default="all-nontest")
    cli.add_argument("--views", type=int, default=24)
    cli.add_argument("--image-size", type=int, default=256)
    cli.add_argument("--distance", type=float, default=2.7)
    cli.add_argument("--elevation", type=float, default=15)
    cli.add_argument("--radius", type=float, default=4)
    cli.add_argument("--sigma", type=float, default=2)
    cli.add_argument("--points-per-pixel", type=int, default=32)
    cli.add_argument("--neighbors", type=int, default=20)
    cli.add_argument("--minimum-points", type=int, default=1024)
    cli.add_argument("--view-batch", type=int, default=4)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--devices", nargs="+", help="Render objects concurrently on these devices")
    cli.add_argument("--renderer", choices=("pytorch3d", "torch"), default="pytorch3d")
    cli.add_argument("--limit", type=int)
    return cli


if __name__ == "__main__":
    torch.set_num_threads(4)
    prepare(parser().parse_args())
