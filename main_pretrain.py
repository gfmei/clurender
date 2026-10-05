"""CluRender joint pretraining. Run --help or --smoke for a complete example."""

import argparse
import datetime
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
from models.encoders import add_octformer_arguments, build_encoder, encoder_checkpoint
from models.pointnet import PointNet
from models.renderer import RenderConfig


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", help="Paired .npz directory; defaults to $CLURENDER_DATA_DIR/paired")
    p.add_argument("--split", help="Text file listing sample paths relative to root")
    p.add_argument("--output", help="Checkpoint directory (default checkpoints/clurender)")
    p.add_argument("--model", choices=("dgcnn", "pointnet", "octformer"), default="dgcnn",
                   help="Encoder; octformer is a batched OctFormer (see models/octformer.py)")
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
    p.add_argument("--render-weight", type=float, default=1.0, help="Weight of the rendering loss (0 removes it)")
    p.add_argument("--cluster-weight", type=float, default=1.0,
                   help="Weight of the clustering loss, including orthogonality (0 removes it)")
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
                   help="Recompute rasterization in backward: less memory, slower (default disabled)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None, help="CPU Torch threads (default 4)")
    p.add_argument("--resume", help="Resume a full training checkpoint, including saved model settings")
    p.add_argument("--save-every", type=int, default=None, help="Checkpoint interval (default 10)")
    p.add_argument("--log-every", type=int, default=None, help="Logging interval (default 10)")
    p.add_argument("--steps-per-epoch", type=int, help="Optional cap for short debugging runs")
    p.add_argument("--smoke", action="store_true", help="Tiny synthetic CPU run; no external data or PyTorch3D needed")
    p.add_argument("--monitor-svm", help="ModelNet40 h5 folder: score a linear SVM on frozen features every "
                   "--monitor-every epochs, fit and validated on a fixed 80/20 split of its train set")
    p.add_argument("--monitor-every", type=int, default=None, help="Epochs between SVM checks (default 5)")
    p.add_argument("--patience", type=int, help="Stop after this many SVM checks without improvement")
    p.add_argument("--min-delta", type=float, default=None,
                   help="Validation accuracy gain that counts as improvement (default 0.002)")
    add_octformer_arguments(p)
    return p


def build_model(args):
    if args.model in ("dgcnn", "octformer"):
        backbone = build_encoder(vars(args))
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
                     sinkhorn_tolerance=args.sinkhorn_tolerance, render_weight=args.render_weight,
                     cluster_weight=args.cluster_weight)


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
                     "checkpoint_rendering", "steps_per_epoch", "epochs", "output", "root", "split",
                     "monitor_svm", "monitor_every", "patience", "min_delta"):
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
                          "save_every": 10, "log_every": 10, "checkpoint_rendering": False,
                          "monitor_every": 5, "min_delta": 0.002}.items():
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
    if args.monitor_every < 1 or not 0 <= args.min_delta < 1 or (args.patience is not None and args.patience < 1):
        raise ValueError("monitor-every and patience must be positive and min-delta in [0, 1)")
    if args.patience is not None and not args.monitor_svm:
        raise ValueError("--patience needs --monitor-svm")
    if min(args.num_points, args.num_views, args.num_clusters, args.image_size,
           args.sinkhorn_iterations, args.points_per_pixel, args.num_projections, *args.color_dims) < 1:
        raise ValueError("Model, sampling and rendering dimensions must be positive")
    if args.emb_dims < 2 or args.seed < 0:
        raise ValueError("emb-dims must be at least 2 and seed must be nonnegative")
    for name in ("lr", "weight_decay", "lr_gamma", "epsilon", "sinkhorn_tolerance", "orthogonal_weight",
                 "render_weight", "cluster_weight", "radius", "sigma", "background", "image_blur"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if (min(args.epsilon, args.radius, args.sigma, args.image_blur) <= 0
            or min(args.sinkhorn_tolerance, args.orthogonal_weight, args.render_weight, args.cluster_weight) < 0
            or args.render_weight == args.cluster_weight == 0):
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


class SvmMonitor:
    """Linear SVM accuracy of frozen encoder features on held-out ModelNet40 training objects.

    A fixed, class-stratified 20% of the ModelNet40 train split is held out for
    validation; the ModelNet40 test split is never used.
    """

    def __init__(self, root, seed, device, num_points=1024):
        from datasets.downstream import load_classification
        from models.finetune import subsample

        points, labels, _ = load_classification("modelnet40", root, "train")
        generator = torch.Generator().manual_seed(seed)
        held_out = torch.zeros(len(labels), dtype=torch.bool)
        for label in labels.unique():
            members = (labels == label).nonzero().flatten()
            held_out[members[torch.randperm(len(members), generator=generator)][:max(1, len(members) // 5)]] = True
        self.points = torch.cat([subsample(batch.to(device), num_points).cpu() for batch in points.split(64)])
        self.labels, self.held_out = labels.numpy(), held_out.numpy()

    @torch.no_grad()
    def score(self, encoder, device):
        from sklearn.svm import SVC

        training = encoder.training
        encoder.eval()
        features = []
        for batch in self.points.split(64):
            per_point = encoder(batch.to(device))[1]
            features.append(torch.cat((per_point.amax(-1), per_point.mean(-1)), 1).cpu())
        encoder.train(training)
        features = torch.cat(features).numpy()
        svm = SVC(C=0.1, kernel="linear").fit(features[~self.held_out], self.labels[~self.held_out])
        return float(svm.score(features[self.held_out], self.labels[self.held_out]))


def early_stopping(state, accuracy, min_delta, patience):
    """Record a validation accuracy in {"best", "stale"}; return (improved, stop)."""
    improved = accuracy > state["best"] + min_delta
    if improved:
        state.update(best=accuracy, stale=0)
    else:
        state["stale"] += 1
    return improved, patience is not None and state["stale"] >= patience


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
    if (checkpoint is not None and args.patience is not None
            and checkpoint.get("monitor", {}).get("stale", 0) >= args.patience):
        print(f"Training stopped early at epoch {checkpoint['epoch']}; increase --patience to continue.", flush=True)
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
        # Long enough for the other processes to wait while the first scores the SVM monitor.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=datetime.timedelta(minutes=60))
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
    monitor_state = {"best": float("-inf"), "stale": 0}
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step, best_loss = checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_loss"]
        monitor_state = checkpoint.get("monitor", monitor_state)
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
    monitor = (SvmMonitor(args.monitor_svm, args.seed, device, args.num_points)
               if args.monitor_svm and rank == 0 else None)
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
        monitor_improved = stop = False
        if args.monitor_svm and (epoch + 1) % args.monitor_every == 0:
            if rank == 0:
                began = time.perf_counter()
                metrics["svm_val_accuracy"] = monitor.score(model.backbone, device)
                metrics["svm_seconds"] = time.perf_counter() - began
                monitor_improved, stop = early_stopping(monitor_state, metrics["svm_val_accuracy"],
                                                        args.min_delta, args.patience)
            if world > 1:
                flag = torch.tensor([int(stop)], device=device)
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
        if rank != 0:
            if stop:
                break
            continue  # Only the first process writes checkpoints and logs.
        state = {"format_version": 2, "epoch": epoch + 1, "global_step": global_step,
                 "best_loss": best_loss, "monitor": monitor_state, "args": vars(args), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_save(state, output / "last.pth")
        atomic_save(encoder_checkpoint(model, args), output / "backbone.pth")
        if improved:
            atomic_save(state, output / "best.pth")
        if monitor_improved:
            atomic_save(state, output / "best_svm.pth")
            atomic_save(encoder_checkpoint(model, args), output / "backbone_best_svm.pth")
        if (epoch + 1) % args.save_every == 0:
            atomic_save(state, output / f"epoch_{epoch + 1:04d}.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if stop:
            print(f"Early stop: SVM validation accuracy did not improve for {args.patience} checks "
                  f"(best {monitor_state['best']:.4f}).", flush=True)
            break
    if world > 1:
        dist.barrier()  # Every process returns after the final checkpoint exists.
        dist.destroy_process_group()
    if rank == 0:
        print(f"Saved checkpoint: {output / 'last.pth'}", flush=True)
    return output / "last.pth"


if __name__ == "__main__":
    train(parser().parse_args())
