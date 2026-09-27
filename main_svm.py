"""Linear SVM on frozen features of the pretrained DGCNN encoder.

The standard evaluation of unsupervised point cloud representations: global
features (max- and average-pooled) of the frozen encoder, from a deterministic
FPS subsample of each cloud, train a linear SVM on the train split, which is
then scored on the test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets.downstream import load_classification
from models.dgcnn import DGCNN
from models.finetune import load_encoder, subsample


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("modelnet40", "scanobjectnn"), default="modelnet40")
    p.add_argument("--root", required=True)
    p.add_argument("--variant", choices=("objbg", "objonly", "hardest"), default="hardest")
    p.add_argument("--pretrained", required=True, help="backbone.pth or a full main_pretrain.py checkpoint")
    p.add_argument("--output", help="Optional results.json path")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--c", type=float, default=0.1, help="SVM regularization")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--emb-dims", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


@torch.no_grad()
def features(encoder, points, args):
    outputs = []
    for batch in points.split(args.batch_size):
        per_point = encoder(subsample(batch.to(args.device), args.num_points))[1]
        outputs.append(torch.cat((per_point.amax(-1), per_point.mean(-1)), 1).cpu())
    return torch.cat(outputs).numpy()


def run(args):
    from sklearn.svm import SVC

    encoder = DGCNN(args.emb_dims, args.k, num_cls=-1)
    load_encoder(encoder, args.pretrained)
    encoder.to(args.device).eval()
    splits = {split: load_classification(args.dataset, args.root, split, variant=args.variant)
              for split in ("train", "test")}
    train_features, test_features = (features(encoder, splits[s][0], args) for s in ("train", "test"))
    train_labels, test_labels = (splits[s][1].numpy() for s in ("train", "test"))
    prediction = SVC(C=args.c, kernel="linear").fit(train_features, train_labels).predict(test_features)
    per_class = [np.mean(prediction[test_labels == c] == c) for c in np.unique(test_labels)]
    results = {"dataset": args.dataset, "variant": args.variant if args.dataset == "scanobjectnn" else None,
               "pretrained": args.pretrained, "c": args.c, "num_points": args.num_points,
               "accuracy": float(np.mean(prediction == test_labels)), "class_accuracy": float(np.mean(per_class))}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    return results


if __name__ == "__main__":
    run(parser().parse_args())
