"""CluRender joint pretraining. Run --help or --smoke for a complete example."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.multiview import MultiViewPointCloudDataset, write_smoke_dataset
from models.clurender import CluRender
from models.dgcnn import DGCNN
from models.pointnet import PointNet
from models.renderer import RenderConfig


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", help="Paired .npz directory; defaults to $CLURENDER_DATA_DIR/paired")
    p.add_argument("--split", help="Text file listing sample paths relative to root")
    p.add_argument("--output", help="Checkpoint directory (default checkpoints/clurender)")
    p.add_argument("--model", choices=("dgcnn", "pointnet"), default="dgcnn")
    p.add_argument("--emb-dims", type=int, default=1024)
    p.add_argument("--k", type=int, default=20, help="DGCNN neighbors")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--num-clusters", type=int, default=64)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--sampling", choices=("fps", "random"), default="fps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, help="Total epochs, including resumed epochs (default 250)")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=20)
    p.add_argument("--lr-gamma", type=float, default=0.7)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--sinkhorn-iterations", type=int, default=2000, help="Maximum clustering Sinkhorn iterations")
    p.add_argument("--sinkhorn-tolerance", type=float, default=0.01,
                   help="Stop clustering Sinkhorn below this relative marginal error (0: run every iteration)")
    p.add_argument("--orthogonal-weight", type=float, default=0.01)
    p.add_argument("--render-weight", type=float, default=1.0, help="Weight of the rendering loss")
    p.add_argument("--round-assignments", action=argparse.BooleanOptionalAction, default=True,
                   help="Enforce clustering marginals after the fixed Sinkhorn budget")
    p.add_argument("--renderer", choices=("pytorch3d", "torch"), default="pytorch3d")
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--points-per-pixel", type=int, default=128)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--image-loss", choices=("sinkhorn", "sliced"), default="sliced",
                   help="sliced: sliced Wasserstein (memory linear in pixels); sinkhorn: entropic OT")
    p.add_argument("--ot-backend", choices=("auto", "tensorized", "online"), default="auto")
    p.add_argument("--image-blur", type=float, default=0.01)
    p.add_argument("--num-projections", type=int, default=128)
    p.add_argument("--color-dims", type=int, nargs=6, default=[512, 256, 128, 128, 64, 32])
    p.add_argument("--workers", type=int, default=None, help="Data-loader workers (default 4)")
    p.add_argument("--device", default=None, help="auto (default), cpu, cuda, or cuda:N")
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=None,
                   help="Single-process DataParallel over visible GPUs; torchrun (one process per GPU) scales better")
    p.add_argument("--checkpoint-rendering", action=argparse.BooleanOptionalAction, default=None,
                   help="Recompute rasterization in backward to reduce view memory (default enabled)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None, help="CPU Torch threads (default 4)")
    p.add_argument("--resume", help="Resume a full training checkpoint, including saved model settings")
    p.add_argument("--save-every", type=int, default=None, help="Checkpoint interval (default 10)")
    p.add_argument("--log-every", type=int, default=None, help="Logging interval (default 10)")
    p.add_argument("--steps-per-epoch", type=int, help="Optional cap for short debugging runs")
    p.add_argument("--smoke", action="store_true", help="Tiny synthetic CPU run; no external data or PyTorch3D needed")
    return p


def build_model(args):
    if args.model == "dgcnn":
        backbone = DGCNN(args.emb_dims, args.k, num_cls=-1)
    else:
        backbone = PointNet(args.emb_dims, feature_transform=True, feat_type="global")
    config = RenderConfig(render_size=args.image_size, points_per_pixel=args.points_per_pixel,
                          radius=args.radius, sigma=args.sigma, backend=args.renderer,
                          background=args.background)
    return CluRender(backbone, dim=args.emb_dims, num_clus=args.num_clusters, render_cfg=config,
                     image_loss=args.image_loss, ot_backend=args.ot_backend, image_blur=args.image_blur,
                     num_projections=args.num_projections, epsilon=args.epsilon,
                     sinkhorn_iterations=args.sinkhorn_iterations, orthogonal_weight=args.orthogonal_weight,
                     color_dims=args.color_dims, round_assignments=args.round_assignments,
                     checkpoint_rendering=args.checkpoint_rendering,
                     sinkhorn_tolerance=args.sinkhorn_tolerance, render_weight=args.render_weight)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resolve_args(args, checkpoint=None):
    generate_smoke = checkpoint is None and args.smoke and args.root is None
    if checkpoint is not None:
        # Restore all objective/architecture settings. Only execution settings
        # and explicitly supplied total epochs/output/data location may change.
        saved = vars(parser().parse_args([]))
        saved.update(checkpoint["args"])
        # Old checkpoints predate marginal rounding. Preserve their objective
        # setting; newly trained runs use feasible assignments by default.
        saved["round_assignments"] = checkpoint["args"].get("round_assignments", False)
        # Likewise, older runs used a fixed Sinkhorn budget without early stopping.
        saved["sinkhorn_tolerance"] = checkpoint["args"].get("sinkhorn_tolerance", 0.0)
        for name in ("device", "workers", "threads", "data_parallel", "save_every", "log_every",
                     "checkpoint_rendering", "steps_per_epoch", "epochs", "output", "root", "split"):
            if getattr(args, name) is not None:
                saved[name] = getattr(args, name)
        saved["resume"] = args.resume
        args = argparse.Namespace(**saved)
    elif args.smoke:
        args.device, args.renderer, args.workers = "cpu", "torch", 0
        args.emb_dims, args.k, args.num_points = 32, 4, 32
        args.num_views, args.num_clusters, args.image_size = 2, 4, 16
        args.color_dims = [32, 24, 16, 16, 12, 8]
        args.batch_size, args.points_per_pixel = 2, 8
        args.radius, args.sigma = 2.0, 1.0
        args.epochs = args.epochs if args.epochs is not None else 2
    for name, default in {"device": "auto", "workers": 4, "threads": 4, "data_parallel": False,
                          "save_every": 10, "log_every": 10, "checkpoint_rendering": True}.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    args.epochs = args.epochs if args.epochs is not None else 250
    args.output = str(Path(args.output or "checkpoints/clurender").resolve())
    if not args.root:
        data_dir = os.environ.get("CLURENDER_DATA_DIR")
        if data_dir:
            args.root = str(Path(data_dir).expanduser() / ("smoke" if args.smoke else "paired"))
        elif args.smoke:
            args.root = str(Path(args.output) / "smoke_data")
    if not args.root:
        raise ValueError("Provide --root, set CLURENDER_DATA_DIR, or use --smoke")
    args.root = str(Path(args.root).resolve())
    if args.split:
        args.split = str(Path(args.split).resolve())
    if args.batch_size < 2 or args.epochs < 1 or args.workers < 0:
        raise ValueError("Training requires batch-size >= 2, epochs >= 1 and workers >= 0")
    if args.k < 1 or (args.model == "dgcnn" and args.k > args.num_points):
        raise ValueError("DGCNN k must be positive and no greater than num-points")
    if min(args.save_every, args.log_every, args.threads, args.lr_step) < 1:
        raise ValueError("Logging, saving, thread and scheduler intervals must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 < args.lr_gamma <= 1:
        raise ValueError("Invalid optimizer or scheduler settings")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("steps-per-epoch must be positive")
    if min(args.num_points, args.num_views, args.num_clusters, args.image_size,
           args.sinkhorn_iterations, args.points_per_pixel, args.num_projections, *args.color_dims) < 1:
        raise ValueError("Model, sampling and rendering dimensions must be positive")
    if args.emb_dims < 2 or args.seed < 0:
        raise ValueError("emb-dims must be at least 2 and seed must be nonnegative")
    for name in ("lr", "weight_decay", "lr_gamma", "epsilon", "sinkhorn_tolerance", "orthogonal_weight",
                 "render_weight", "radius", "sigma", "background", "image_blur"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if (min(args.epsilon, args.radius, args.sigma, args.image_blur) <= 0
            or min(args.sinkhorn_tolerance, args.orthogonal_weight, args.render_weight) < 0):
        raise ValueError("Invalid transport, splat, regularization or loss-weight settings")
    if not 0 <= args.background <= 1:
        raise ValueError("background must be in [0, 1]")
    args.generate_smoke_data = generate_smoke
    return args


def reduce_parallel_losses(losses):
    """Average replica means by their sample counts, including uneven shards."""
    counts = losses["batch_size"]
    weights = counts / counts.sum()
    return {name: value.max() if name == "assignment_error" else (value * weights).sum()
            for name, value in losses.items() if name != "batch_size"}


def distributed_context():
    """(rank, world size, local rank) when launched by torchrun, else (0, 1, 0)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    return int(os.environ["RANK"]), world, int(os.environ["LOCAL_RANK"])


def average_across_processes(losses, world):
    """Mean of every loss over processes; the maximum for assignment_error."""
    if world == 1:
        return losses
    names = sorted(losses)
    values = torch.stack([losses[name].detach().float() for name in names])
    maximum = values.clone()
    dist.all_reduce(values)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {name: maximum[i] if name == "assignment_error" else values[i] / world for i, name in enumerate(names)}


def validate_parallel_batch(batch_size, num_devices, model):
    # DataParallel uses torch.chunk, whose last shard can be smaller than
    # floor(batch_size / num_devices), e.g. 10 samples / 4 GPUs -> 3,3,3,1.
    chunk_size = math.ceil(batch_size / num_devices)
    smallest = batch_size % chunk_size or chunk_size
    if model == "pointnet" and smallest < 2:
        raise ValueError("PointNet requires at least two samples in every GPU shard for batch normalization")


def train(args):
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    args = resolve_args(args, checkpoint)
    if checkpoint is not None and checkpoint["epoch"] >= args.epochs:
        # Do not rewrite config.json or return a nonexistent output checkpoint
        # when the requested training is already complete.
        print(f"Checkpoint already completed {checkpoint['epoch']} epochs; increase --epochs to continue.", flush=True)
        return Path(args.resume).resolve()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank = distributed_context()
    # Resolve "auto" locally: checkpoints keep it, so a resumed run can change node type.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if world > 1:
        # torchrun: one process per GPU, each with 1/world of every batch.
        if args.data_parallel:
            raise ValueError("Use either torchrun (distributed) or --data-parallel, not both")
        if args.batch_size % world:
            raise ValueError(f"batch-size {args.batch_size} is not divisible by {world} processes")
        validate_parallel_batch(args.batch_size, world, args.model)
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if args.data_parallel and (device.type != "cuda" or device.index not in (None, 0)):
        raise ValueError("--data-parallel requires --device cuda or cuda:0")
    if args.data_parallel:
        validate_parallel_batch(args.batch_size, torch.cuda.device_count(), args.model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.generate_smoke_data and not Path(args.root).exists() and rank == 0:
        write_smoke_dataset(args.root, num_points=args.num_points, image_size=args.image_size,
                            num_views=args.num_views)
    if world > 1:
        dist.barrier()
    dataset = MultiViewPointCloudDataset(args.root, args.num_points, args.num_views,
                                        args.image_size, args.split, args.sampling, args.seed)
    if len(dataset) < args.batch_size:
        raise ValueError(f"Dataset has {len(dataset)} samples, less than batch-size {args.batch_size}")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (DistributedSampler(dataset, world, rank, shuffle=True, seed=args.seed, drop_last=True)
               if world > 1 else None)
    loader = DataLoader(dataset, batch_size=args.batch_size // world, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda",
                        generator=generator)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    start_epoch, global_step, best_loss = 0, 0, float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step, best_loss = checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_loss"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            for index, state in enumerate(checkpoint["cuda_rng"][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, index)
    if world > 1:
        training_model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    else:
        training_model = torch.nn.DataParallel(model) if args.data_parallel else model
    if rank == 0:
        (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print(f"CluRender: {len(dataset)} samples, {world} x {device.type}, {args.image_loss} fitting, "
              f"{args.renderer} renderer", flush=True)
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        training_model.train()
        totals, seen, began = {}, 0, time.perf_counter()
        for step, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()
                     if isinstance(value, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            local = reduce_parallel_losses(training_model(**batch, return_details=True))
            # Identical on every process, so all of them stop together on failure.
            losses = average_across_processes(local, world)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, step {step + 1}")
            local["loss"].backward()
            # Error out before saving corrupted parameters if gradients fail.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            seen += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item()
            if step % args.log_every == 0 and rank == 0:
                print(f"epoch={epoch + 1} step={step + 1} loss={losses['loss'].item():.6f} "
                      f"clustering={losses['clustering'].item():.6f} rendering={losses['rendering'].item():.6f}", flush=True)
            if args.steps_per_epoch is not None and seen >= args.steps_per_epoch:
                break
        metrics = {name: value / seen for name, value in totals.items()}
        metrics.update(epoch=epoch + 1, global_step=global_step, lr=optimizer.param_groups[0]["lr"],
                       seconds=time.perf_counter() - began)
        scheduler.step()
        improved = metrics["loss"] < best_loss
        best_loss = min(best_loss, metrics["loss"])
        if rank != 0:
            continue  # Only the first process writes checkpoints and logs.
        state = {"format_version": 2, "epoch": epoch + 1, "global_step": global_step,
                 "best_loss": best_loss, "args": vars(args), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_save(state, output / "last.pth")
        atomic_save(model.backbone.state_dict(), output / "backbone.pth")
        if improved:
            atomic_save(state, output / "best.pth")
        if (epoch + 1) % args.save_every == 0:
            atomic_save(state, output / f"epoch_{epoch + 1:04d}.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    if world > 1:
        dist.barrier()  # Every process returns after the final checkpoint exists.
        dist.destroy_process_group()
    if rank == 0:
        print(f"Saved checkpoint: {output / 'last.pth'}", flush=True)
    return output / "last.pth"


if __name__ == "__main__":
    train(parser().parse_args())
