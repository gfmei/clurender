"""Fine-tune the pretrained DGCNN encoder for ShapeNetPart part segmentation.

Evaluation follows Point-MAE's segmentation/main.py: predictions are limited
to the parts of each object's category, a part absent from both prediction and
ground truth scores IoU 1, and accuracy, class mIoU and instance mIoU are
computed on the test split after every epoch. Point-MAE reports the best
instance mIoU over epochs (selected on the test set); results.json also
records the last epoch, which involves no such selection.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.shapenetpart import CATEGORIES, PartSegMetrics, ShapeNetPartText
from main_finetune_cls import atomic_save
from models.finetune import (DGCNNPartSegmenter, augment, build_optimizer, load_encoder, part_mask,
                             set_encoder_frozen)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True,
                   help="shapenetcore_partanno_segmentation_benchmark_v0_normal or PartAnnotation directory")
    p.add_argument("--exclude-overlap", action="store_true",
                   help="Drop train/val objects that repeat or also appear in test, instead of refusing such splits")
    p.add_argument("--pretrained", help="backbone.pth or a full main_pretrain.py checkpoint; omit to train from scratch")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pth")
    p.add_argument("--label-fraction", type=float, default=1.0,
                   help="Train on this class-stratified fraction of train+val objects, repeated so that an "
                        "epoch has about as many samples as a full-data epoch")
    p.add_argument("--label-seed", type=int, default=0, help="Seed that selects the labeled subset")
    p.add_argument("--freeze-encoder-epochs", type=int, default=0,
                   help="Train only the head for this many epochs first (linear probe, then fine-tune)")
    p.add_argument("--encoder-lr-scale", type=float, default=1.0, help="Encoder learning rate relative to --lr")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--k", type=int, default=40, help="DGCNN neighbors (40 is DGCNN's part segmentation setting)")
    p.add_argument("--emb-dims", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


@torch.no_grad()
def evaluate(model, loader, mask, device):
    model.eval()
    metrics = PartSegMetrics()
    for batch in loader:
        categories = batch["category"].to(device)
        logits = model(batch["points"].to(device), categories).transpose(1, 2)
        prediction = logits.masked_fill(~mask[categories, None, :], -torch.inf).argmax(-1)
        metrics.update(prediction.cpu().numpy(), batch["parts"].numpy(), batch["category"].numpy())
    # Plain Python numbers: checkpoints are read back with weights_only=True.
    return json.loads(json.dumps(metrics.compute()))


def run(args):
    if args.num_points < 1 or args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        raise ValueError("num-points and epochs must be positive, batch-size at least 2, workers nonnegative")
    if not 0 <= args.freeze_encoder_epochs <= args.epochs or not args.encoder_lr_scale > 0:
        raise ValueError("freeze-encoder-epochs must lie in [0, epochs] and encoder-lr-scale be positive")
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = ShapeNetPartText(args.root, "trainval", args.num_points, seed=args.seed + 1,
                             exclude_overlap=args.exclude_overlap, label_fraction=args.label_fraction,
                             label_seed=args.label_seed)
    test = ShapeNetPartText(args.root, "test", args.num_points, seed=args.seed, exclude_overlap=args.exclude_overlap)
    (output / "split_audit.json").write_text(json.dumps(train.split_audit, indent=2) + "\n")
    generator = torch.Generator().manual_seed(args.seed)
    pin = args.device.startswith("cuda")
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = DGCNNPartSegmenter(emb_dims=args.emb_dims, k=args.k, dropout=args.dropout)
    loaded = load_encoder(model.encoder, args.pretrained) if args.pretrained else 0
    model.to(args.device)
    mask = part_mask(CATEGORIES, args.device)
    optimizer, scheduler = build_optimizer(model, args)
    start, best = 0, {"instance_miou": -1.0}
    if args.resume:
        state = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start, best = state["epoch"], state["best"]
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"ShapeNetPart: {len(train.ids)} train+val objects ({len(train)} samples per epoch), {len(test)} test; "
          f"{'pretrained encoder (' + str(loaded) + ' tensors)' if loaded else 'random initialization'}", flush=True)

    for epoch in range(start, args.epochs):
        # A different point subsample of every training object in each epoch.
        train.seed = args.seed + 1 + epoch
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=pin, generator=generator)
        model.train()
        frozen = epoch < args.freeze_encoder_epochs
        set_encoder_frozen(model, frozen)
        began, seen, loss_sum = time.perf_counter(), 0, 0.0
        for batch in loader:
            points = augment(batch["points"].to(args.device), generator)
            parts, categories = batch["parts"].to(args.device), batch["category"].to(args.device)
            loss = F.cross_entropy(model(points, categories), parts, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(parts)
            loss_sum += loss.item() * len(parts)
        scheduler.step()
        metrics = evaluate(model, test_loader, mask, args.device)
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "encoder_frozen": frozen,
                  "train_loss": loss_sum / seen,
                  "test_accuracy": metrics["overall_accuracy"], "test_class_miou": metrics["class_miou"],
                  "test_instance_miou": metrics["instance_miou"], "seconds": time.perf_counter() - began}
        if metrics["instance_miou"] > best["instance_miou"]:
            best = {"epoch": epoch + 1, **metrics}
            atomic_save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, output / "best.pth")
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                     "torch_rng": torch.get_rng_state(), "epoch": epoch + 1, "best": best, "last": metrics,
                     "args": vars(args)}, output / "last.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    last = torch.load(output / "last.pth", map_location="cpu", weights_only=True)["last"]
    results = {"pretrained": args.pretrained, "epochs": args.epochs, "label_fraction": args.label_fraction,
               "label_seed": args.label_seed, "seed": args.seed, "last_epoch": last,
               "best_epoch_test_selected": best}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"last_instance_miou": last["instance_miou"], "last_class_miou": last["class_miou"],
                      "best_instance_miou": best["instance_miou"], "best_epoch": best["epoch"]}), flush=True)
    return results


if __name__ == "__main__":
    run(parser().parse_args())
