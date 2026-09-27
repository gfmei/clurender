"""Validate CluRender on PyTorch3D's public cow mesh, without ShapeNet access.

The two objects are geometry variants of one tutorial mesh, not research data.
They use a ShapeNetCore v2 directory layout to exercise the preparation path.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import urlopen
import warnings


ASSETS = {
    "cow.obj": "62337f06b4aa4f2eaa95f854553a02039d853365a78b0a581c26ece5691a6d99",
    "cow.mtl": "ce6398d52486f81f2c329b1683ad4fedce970471aa36e589e05c5177432db6cb",
    "cow_texture.png": "cddabbae52a666173e7953e238b88340d285044dc20b36f8ed3f1a41db534fa5",
}
SOURCE = "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/"


def prepare_fixture(root):
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in ASSETS.items():
        path = source / name
        if not path.exists():
            with urlopen(SOURCE + name, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Tutorial asset checksum mismatch: {name}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            os.replace(temporary, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Tutorial asset checksum mismatch: {path}")
        records.append({"file": name, "url": SOURCE + name, "sha256": expected})
    # A valid synset is needed by PyTorch3D's ShapeNetCore loader. This slot is
    # only a layout fixture: these are cows, not examples of the chair class.
    identifiers = ["03001627/tutorial_cow", "03001627/tutorial_cow_stretched"]
    for identifier, scale in zip(identifiers, [(1, 1, 1), (1.3, .8, 1)]):
        destination = root / "meshes" / identifier / "models"
        destination.mkdir(parents=True, exist_ok=True)
        lines = []
        for line in (source / "cow.obj").read_text().splitlines():
            if line.startswith("v "):
                xyz = [float(value) * factor for value, factor in zip(line.split()[1:4], scale)]
                line = "v " + " ".join(f"{value:.9g}" for value in xyz)
            lines.append(line)
        (destination / "model_normalized.obj").write_text("\n".join(lines) + "\n")
        for name in ("cow.mtl", "cow_texture.png"):
            shutil.copyfile(source / name, destination / name)
    (root / "mesh_ids.txt").write_text("\n".join(identifiers) + "\n")
    (root / "train.txt").write_text("\n".join(name + ".npz" for name in identifiers) + "\n")
    (root / "source.json").write_text(json.dumps({
        "purpose": "Code validation only; two variants of the tutorial cow, not ShapeNet data",
        "synset": "03001627 is a directory-layout placeholder, not a class label",
        "files": records,
    }, indent=2) + "\n")


def validate(args):
    import numpy as np
    import torch
    from PIL import Image
    from pytorch3d.datasets import ShapeNetCore, collate_batched_meshes
    from torch.utils.data import DataLoader

    from datasets.multiview import MultiViewPointCloudDataset
    from main_pretrain import build_model, parser, train
    from prepare_shapenet import prepare

    root, output = args.root, args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty --output directory to keep validation runs independent")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA validation requires a GPU allocation")
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The following categories are included in ShapeNetCore")
        meshes = ShapeNetCore(str(root / "meshes"), synsets=["03001627"], version=2)
    mesh_batch = next(iter(DataLoader(meshes, batch_size=2, collate_fn=collate_batched_meshes)))
    assert len(mesh_batch["mesh"]) == 2
    prepare(argparse.Namespace(
        root=str(root / "meshes"), output=str(root / "paired"), split=str(root / "mesh_ids.txt"),
        views=8, num_points=2048, image_size=256, distance=2.7, elevation=15,
        background=0., texture_atlas_size=8, device=args.device, seed=0, limit=None, shard=None, overwrite=True))
    report = {"purpose": "Code validation, not ShapeNet pretraining", "objects": 2,
              "source": SOURCE, "device": args.device, "torch": torch.__version__,
              "training_image_size": args.image_size, "backbones": {}}
    for path in sorted((root / "paired").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            images = archive["images"]
            coverage = (images.max(-1) > 0).mean((1, 2))
            assert (coverage > .01).all() and (coverage < .9).all(), (path, coverage)
            assert images.std() > 1, "Missing texture variation"
            Image.fromarray(np.concatenate(list(images), axis=1)).save(output / f"{path.stem}-views.png")

    for backbone in ("dgcnn", "pointnet"):
        run = output / backbone
        command = ["--root", str(root / "paired"), "--split", str(root / "train.txt"),
                   "--output", str(run), "--model", backbone, "--device", args.device,
                   "--batch-size", "2", "--epochs", "2", "--num-points", "1024", "--num-views", "8",
                   "--image-size", str(args.image_size), "--ot-backend", "online",
                   "--workers", "0", "--threads", "4", "--log-every", "1"]
        checkpoint_path = train(parser().parse_args(command))
        before = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        train(parser().parse_args(["--resume", str(checkpoint_path), "--epochs", "3", "--device", args.device]))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert before["global_step"] == 2 and state["global_step"] == 3 and state["epoch"] == 3
        assert any(not torch.equal(state["model"][key], before["model"][key])
                   for key in state["model"] if state["model"][key].is_floating_point())
        assert all(torch.isfinite(value).all() for value in state["model"].values())
        exported = torch.load(run / "backbone.pth", map_location="cpu", weights_only=True)
        model = build_model(argparse.Namespace(**state["args"])).to(args.device).eval()
        model.load_state_dict(state["model"])
        for key, value in exported.items():
            torch.testing.assert_close(value, state["model"]["backbone." + key], rtol=0, atol=0)
        data = MultiViewPointCloudDataset(root / "paired", 1024, 8, args.image_size, root / "train.txt")
        batch = next(iter(DataLoader(data, batch_size=2)))
        batch = {key: value.to(args.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        losses = model(**batch, return_details=True)
        # Isolate rendering: clustering gradients must not hide a broken RGB path.
        losses["rendering"].backward()
        gradient_norms = {}
        for name in ("backbone", "color"):
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = torch.stack([g.norm() for g in gradients]).norm().item()
            assert norm > 0, f"Rendering did not reach {backbone}/{name}"
            gradient_norms[name] = norm
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [row["epoch"] for row in metrics] == [1, 2, 3]
        report["backbones"][backbone] = {"metrics": metrics, "rendering_gradient_norms": gradient_norms,
                                          "resume_and_export": "passed"}
        del model, losses, batch, state, before
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
    report["elapsed_seconds"] = time.perf_counter() - start
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Tutorial validation passed: {output / 'validation.json'}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True, help="Directory for downloaded assets and paired fixtures")
    cli.add_argument("--output", type=Path, help="Empty directory for checkpoints and validation.json")
    cli.add_argument("--prepare-only", action="store_true", help="Download and arrange assets; no PyTorch3D required")
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--image-size", type=int, default=64, help="Training resolution; mesh targets are rendered at 256")
    args = cli.parse_args()
    if not args.prepare_only and args.output is None:
        cli.error("--output is required unless --prepare-only is supplied")
    prepare_fixture(args.root)
    if not args.prepare_only:
        validate(args)
