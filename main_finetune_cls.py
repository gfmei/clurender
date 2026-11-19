"""Fine-tune the pretrained encoder for point cloud classification.

Datasets: ModelNet40, ScanObjectNN (objbg, objonly, hardest), or one few-shot
ModelNet40 episode. Evaluation follows Point-MAE's runner_finetune.py: after
every epoch the test split is classified from an FPS subsample, the checkpoint
with the best test accuracy is kept, and that checkpoint is finally tested
again with votes that average the logits of randomly scaled and translated
copies. Point-MAE reports this best (test-selected) accuracy; results.json
also records the last epoch, which involves no selection on the test set.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from datasets.downstream import load_classification
from models.encoders import ENCODERS, add_octformer_arguments, downstream_encoder
from models.finetune import DGCNNClassifier, augment, build_optimizer, load_encoder, subsample


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("modelnet40", "scanobjectnn", "fewshot"), required=True)
    p.add_argument("--root", required=True, help="ModelNet40 h5 folder, ScanObjectNN root or ModelNetFewshot root")
    p.add_argument("--variant", choices=("objbg", "objonly", "hardest"), default="hardest",
                   help="ScanObjectNN variant")
    p.add_argument("--way", type=int, default=5)
    p.add_argument("--shot", type=int, default=10)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--pretrained", help="backbone.pth or a full main_pretrain.py checkpoint; omit to train from scratch")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pth")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--point-pool", type=int, default=1200,
                   help="Training FPS pool from which num-points are drawn at random (Point-MAE: 1200 for ModelNet40)")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--k", type=int, default=20, help="DGCNN neighbors")
    p.add_argument("--emb-dims", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--votes", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=ENCODERS, default="dgcnn",
                   help="Encoder when training from scratch; a full pretraining checkpoint sets it")
    add_octformer_arguments(p)
    return p


def accuracy(logits, labels, num_classes):
    prediction = logits.argmax(-1)
    per_class = [(prediction[labels == c] == c).float().mean() for c in range(num_classes) if (labels == c).any()]
    return {"accuracy": (prediction == labels).float().mean().item(),
            "class_accuracy": torch.stack(per_class).mean().item()}


@torch.no_grad()
def predict(model, points, args, votes=0, generator=None):
    """Logits for points already subsampled to num_points, or averaged votes."""
    model.eval()
    outputs = []
    for batch in points.split(args.batch_size):
        batch = batch.to(args.device)
        if not votes:
            outputs.append(model(batch).cpu())
            continue
        copies = [model(augment(subsample(batch, args.num_points, args.point_pool, generator), generator))
                  for _ in range(votes)]
        outputs.append(torch.stack(copies).mean(0).cpu())
    return torch.cat(outputs)


def atomic_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(args):
    if args.num_points < 1 or args.epochs < 1 or args.batch_size < 2 or args.votes < 1:
        raise ValueError("num-points and epochs must be positive, batch-size at least 2, votes positive")
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    options = dict(variant=args.variant, way=args.way, shot=args.shot, fold=args.fold)
    train_points, train_labels, num_classes = load_classification(args.dataset, args.root, "train", **options)
    test_points, test_labels, _ = load_classification(args.dataset, args.root, "test", **options)
    if len(train_points) < args.batch_size:
        raise ValueError(f"Training split has {len(train_points)} samples, fewer than one batch")
    # Deterministic FPS of the test split, computed once.
    test_input = torch.cat([subsample(batch.to(args.device), args.num_points).cpu()
                            for batch in test_points.split(args.batch_size)])

    model = DGCNNClassifier(num_classes, dropout=args.dropout, encoder=downstream_encoder(args))
    loaded = load_encoder(model.encoder, args.pretrained) if args.pretrained else 0
    model.to(args.device)
    optimizer, scheduler = build_optimizer(model, args)
    generator = torch.Generator().manual_seed(args.seed)
    start, best = 0, {"accuracy": -1.0}
    if args.resume:
        state = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start, best = state["epoch"], state["best"]
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"{args.dataset}: {len(train_points)} train, {len(test_points)} test, {num_classes} classes; "
          f"{'pretrained encoder (' + str(loaded) + ' tensors)' if loaded else 'random initialization'}", flush=True)

    for epoch in range(start, args.epochs):
        model.train()
        began, seen, loss_sum, correct = time.perf_counter(), 0, 0.0, 0
        order = torch.randperm(len(train_points), generator=generator)
        for indices in order.split(args.batch_size):
            if len(indices) < args.batch_size:
                break  # Batch normalization needs full batches; Point-MAE also drops the last one.
            points = subsample(train_points[indices].to(args.device), args.num_points, args.point_pool, generator)
            labels = train_labels[indices].to(args.device)
            logits = model(augment(points, generator))
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
        scheduler.step()
        test = accuracy(predict(model, test_input, args), test_labels, num_classes)
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / seen,
                  "train_accuracy": correct / seen, **{"test_" + k: v for k, v in test.items()},
                  "seconds": time.perf_counter() - began}
        if test["accuracy"] > best["accuracy"]:
            best = {"epoch": epoch + 1, **test}
            atomic_save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, output / "best.pth")
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                     "torch_rng": torch.get_rng_state(), "epoch": epoch + 1, "best": best, "last": record,
                     "args": vars(args)}, output / "last.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    last = torch.load(output / "last.pth", map_location="cpu", weights_only=True)["last"]
    model.load_state_dict(torch.load(output / "best.pth", map_location="cpu", weights_only=True)["model"])
    vote = accuracy(predict(model, test_points, args, args.votes, torch.Generator().manual_seed(args.seed)),
                    test_labels, num_classes)
    results = {"dataset": args.dataset, "variant": args.variant if args.dataset == "scanobjectnn" else None,
               "episode": {"way": args.way, "shot": args.shot, "fold": args.fold} if args.dataset == "fewshot" else None,
               "pretrained": args.pretrained, "epochs": args.epochs,
               "last_epoch": {"accuracy": last["test_accuracy"], "class_accuracy": last["test_class_accuracy"]},
               "best_epoch_test_selected": best,
               "best_epoch_vote": {"votes": args.votes, **vote}}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    return results


if __name__ == "__main__":
    run(parser().parse_args())
