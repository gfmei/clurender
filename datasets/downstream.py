"""Downstream classification data: ModelNet40, ScanObjectNN and few-shot ModelNet40.

Every loader returns all samples of a split in memory, as float32 points
(M, 3, P) normalized to the unit sphere and int64 labels (M,). Training
scripts subsample the P points (farthest point sampling, as in Point-MAE).
"""

import pickle
from pathlib import Path

import numpy as np
import torch

# ScanObjectNN variants as in Point-MAE: (directory, file suffix).
SCANOBJECTNN = {"objbg": ("main_split", "objectdataset"),
                "objonly": ("main_split_nobg", "objectdataset"),
                "hardest": ("main_split", "objectdataset_augmentedrot_scale75")}
FEWSHOT_SETTINGS = ((5, 10), (5, 20), (10, 10), (10, 20))


def normalize(points):
    points = points - points.mean(1, keepdims=True)
    return points / np.linalg.norm(points, axis=-1).max(1)[:, None, None].clip(1e-12)


def _read_h5(paths):
    import h5py

    points, labels = [], []
    for path in paths:
        with h5py.File(path, "r") as archive:
            points.append(archive["data"][:].astype(np.float32))
            labels.append(archive["label"][:].reshape(-1).astype(np.int64))
    return np.concatenate(points), np.concatenate(labels)


def _as_tensors(points, labels):
    points = normalize(points[..., :3])
    return torch.from_numpy(np.ascontiguousarray(points.transpose(0, 2, 1))), torch.from_numpy(labels)


def load_classification(dataset, root, split, variant="hardest", way=5, shot=10, fold=0):
    """Return (points, labels, number of classes) for one split ("train" or "test")."""
    root = Path(root)
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if dataset == "modelnet40":
        paths = sorted(root.glob(f"ply_data_{split}*.h5"))
        if not paths:
            raise FileNotFoundError(f"No ModelNet40 ply_data_{split}*.h5 files in {root}")
        return (*_as_tensors(*_read_h5(paths)), 40)
    if dataset == "scanobjectnn":
        if variant not in SCANOBJECTNN:
            raise ValueError(f"ScanObjectNN variant must be one of {sorted(SCANOBJECTNN)}")
        directory, suffix = SCANOBJECTNN[variant]
        prefix = "training" if split == "train" else "test"
        path = root / directory / f"{prefix}_{suffix}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ScanObjectNN file {path}")
        return (*_as_tensors(*_read_h5([path])), 15)
    if dataset == "fewshot":
        if (way, shot) not in FEWSHOT_SETTINGS or not 0 <= fold < 10:
            raise ValueError("Few-shot settings are 5 or 10 way, 10 or 20 shot, folds 0-9")
        # Point-BERT's release and Point-MAE's loader name the folders differently.
        candidates = [root / f"{way}way{shot}shot" / f"{fold}.pkl", root / f"{way}way_{shot}shot" / f"{fold}.pkl"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing few-shot split {candidates[0]}")
        with path.open("rb") as stream:
            items = pickle.load(stream)[split]
        points = np.stack([np.asarray(item[0], dtype=np.float32)[:, :3] for item in items])
        labels = np.array([int(item[1]) for item in items], dtype=np.int64)
        return (*_as_tensors(points, labels), way)
    raise ValueError("dataset must be modelnet40, scanobjectnn or fewshot")
