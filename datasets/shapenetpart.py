"""ShapeNetPart normal text or original PartAnnotation point/label files."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


CATEGORIES = (
    ("Airplane", "02691156", 0, 4), ("Bag", "02773838", 4, 2),
    ("Cap", "02954340", 6, 2), ("Car", "02958343", 8, 4),
    ("Chair", "03001627", 12, 4), ("Earphone", "03261776", 16, 3),
    ("Guitar", "03467517", 19, 3), ("Knife", "03624134", 22, 2),
    ("Lamp", "03636649", 24, 4), ("Laptop", "03642806", 28, 2),
    ("Motorbike", "03790512", 30, 6), ("Mug", "03797390", 36, 2),
    ("Pistol", "03948459", 38, 3), ("Rocket", "04099429", 41, 3),
    ("Skateboard", "04225987", 44, 3), ("Table", "04379243", 47, 3),
)


def split_ids(root, allow_overlap=False):
    result = {}
    for split in ("train", "val", "test"):
        path = Path(root) / "train_test_split" / f"shuffled_{split}_file_list.json"
        ids = ["/".join(value.split("/")[-2:]) for value in json.loads(path.read_text())]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate objects in {split} split")
        if any(len(value.split("/")) != 2 or ".." in value.split("/") for value in ids):
            raise ValueError("Invalid ShapeNetPart object identifier")
        result[split] = ids
    if not allow_overlap and (set(result["train"]) & set(result["val"])
                              or set(result["train"]) & set(result["test"])
                              or set(result["val"]) & set(result["test"])):
        raise ValueError("ShapeNetPart train, validation and test splits overlap")
    return result


def stratified_subset(categories, fraction, seed):
    """Sorted indices keeping round(fraction * n), at least one, objects of each category."""
    rng = np.random.default_rng(seed)
    keep = []
    for category in np.unique(categories):
        members = np.flatnonzero(categories == category)
        keep.extend(rng.choice(members, max(1, round(fraction * len(members))), replace=False))
    return np.sort(np.asarray(keep, dtype=np.int64))


class ShapeNetPartText(Dataset):
    """ShapeNetPart objects with fixed-size point samples.

    With ``label_fraction`` < 1, only a class-stratified subset of objects keeps
    its labels, and the subset is repeated ceil(1 / label_fraction) times so
    that an epoch has about as many samples as a full-data epoch. Each repeat
    draws a different point sample.
    """

    def __init__(self, root, split="trainval", num_points=2048, seed=0, exclude_overlap=False,
                 label_fraction=1.0, label_seed=0):
        if split not in ("train", "val", "trainval", "test") or num_points < 1 or seed < 0:
            raise ValueError("Invalid split, point count or seed")
        if not 0 < label_fraction <= 1 or label_seed < 0:
            raise ValueError("label_fraction must be in (0, 1] and label_seed nonnegative")
        self.root = Path(root).resolve()
        splits = split_ids(self.root, allow_overlap=exclude_overlap)
        self.split_audit = {"original_counts": {key: len(value) for key, value in splits.items()},
                            "train_test_overlap": sorted(set(splits["train"]) & set(splits["test"])),
                            "val_test_overlap": sorted(set(splits["val"]) & set(splits["test"])),
                            "train_val_overlap": sorted(set(splits["train"]) & set(splits["val"]))}
        if exclude_overlap:
            held_out = set(splits["test"])
            splits["train"] = [name for name in splits["train"] if name not in held_out]
            seen = held_out | set(splits["train"])
            splits["val"] = [name for name in splits["val"] if name not in seen]
        self.split_audit["effective_counts"] = {key: len(value) for key, value in splits.items()}
        self.ids = splits["train"] + splits["val"] if split == "trainval" else splits[split]
        if not self.ids:
            raise ValueError(f"Empty {split} split")
        self.num_points, self.seed = num_points, seed
        synsets = {item[1]: i for i, item in enumerate(CATEGORIES)}
        self.categories = np.array([synsets[item.split("/")[0]] for item in self.ids], dtype=np.int64)
        self.repeats = 1
        if label_fraction < 1:
            keep = stratified_subset(self.categories, label_fraction, label_seed)
            self.ids, self.categories = [self.ids[i] for i in keep], self.categories[keep]
            self.repeats = math.ceil(1 / label_fraction)
            self.split_audit.update(label_fraction=label_fraction, label_seed=label_seed,
                                    labeled_objects=len(self.ids), repeats=self.repeats)
        self.paths = [self.root / (item + ".txt") for item in self.ids]
        self.label_paths = None
        if not self.paths[0].exists() and (self.root / self.ids[0].split("/")[0] / "points").is_dir():
            self.paths = [self.root / synset / "points" / (name + ".pts")
                          for synset, name in (item.split("/") for item in self.ids)]
            self.label_paths = [self.root / synset / "expert_verified" / "points_label" / (name + ".seg")
                                for synset, name in (item.split("/") for item in self.ids)]
        records = [(name, path.stat().st_size, path.stat().st_mtime_ns)
                   for name, path in zip(self.ids, self.paths)]
        if self.label_paths is not None:
            # Labels are separate files; their changes must invalidate feature caches.
            records += [("local-one-based-labels:" + name, path.stat().st_size, path.stat().st_mtime_ns)
                        for name, path in zip(self.ids, self.label_paths)]
        description = {"root": str(self.root), "records": records, "points": num_points, "seed": seed,
                       "sampling": "uniform-random-fixed-per-object", "normalization": "center-unit-radius"}
        if label_fraction < 1:
            description.update(label_fraction=label_fraction, label_seed=label_seed)
        self.fingerprint = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.ids) * self.repeats

    def __getitem__(self, index):
        repeat, index = divmod(index, len(self.ids))
        raw = np.loadtxt(self.paths[index], dtype=np.float32, ndmin=2)
        columns = 3 if self.label_paths is not None else 7
        if raw.shape[1] != columns or not len(raw) or not np.isfinite(raw).all():
            raise ValueError(f"Expected {columns} finite columns: {self.paths[index]}")
        category = int(self.categories[index])
        start, count = CATEGORIES[category][2:]
        if self.label_paths is None:
            labels = raw[:, -1]
        else:
            labels = np.loadtxt(self.label_paths[index], dtype=np.float32, ndmin=1)
            if labels.shape != (len(raw),):
                raise ValueError(f"Point/label count mismatch: {self.paths[index]}")
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError(f"Part labels must be finite integers: {self.paths[index]}")
        lower = 1 if self.label_paths is not None else start
        if labels.min() < lower or labels.max() >= lower + count:
            raise ValueError(f"Part labels disagree with object category: {self.paths[index]}")
        parts = labels.astype(np.int64)
        if self.label_paths is not None:
            # Original expert labels are 1-based within each object category.
            parts += start - 1
        points = raw[:, :3].copy()
        points -= points.mean(0)
        radius = np.linalg.norm(points, axis=1).max()
        if radius <= 0:
            raise ValueError(f"Degenerate point cloud: {self.paths[index]}")
        points /= radius
        # Stable across processes, worker counts, split combinations and root locations.
        object_seed = int.from_bytes(hashlib.sha256(self.ids[index].encode()).digest()[:8], "little")
        # The first repeat keeps the full-data sample, so label_fraction=1 is unchanged.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, object_seed] + ([repeat] if repeat else [])))
        choice = rng.choice(len(points), self.num_points, replace=len(points) < self.num_points)
        return {"points": points[choice].T.copy(), "parts": parts[choice], "category": category}


class PartSegMetrics:
    """Instance mIoU, category mIoU and point accuracy, with absent-part IoU=1."""
    def __init__(self):
        self.correct, self.total = 0, 0
        self.iou_sum = np.zeros(16, dtype=np.float64)
        self.shapes = np.zeros(16, dtype=np.int64)

    def update(self, prediction, target, categories):
        prediction, target = np.asarray(prediction), np.asarray(target)
        categories = np.asarray(categories).reshape(-1)
        if prediction.shape != target.shape or target.ndim != 2 or len(categories) != len(target):
            raise ValueError("Expected matching B x N predictions/labels and B categories")
        if not np.isin(categories, np.arange(16)).all():
            raise ValueError("Unknown object category")
        self.correct += int((prediction == target).sum())
        self.total += target.size
        for pred, truth, category in zip(prediction, target, categories):
            start, count = CATEGORIES[category][2:]
            if not np.isin(truth, np.arange(start, start + count)).all():
                raise ValueError("Ground-truth part is invalid for the category")
            ious = []
            for part in range(start, start + count):
                intersection = np.logical_and(pred == part, truth == part).sum()
                union = np.logical_or(pred == part, truth == part).sum()
                ious.append(intersection / union if union else 1.0)
            self.iou_sum[category] += np.mean(ious)
            self.shapes[category] += 1

    def compute(self):
        present = self.shapes > 0
        if not self.total:
            raise ValueError("No predictions to evaluate")
        per_class = self.iou_sum[present] / self.shapes[present]
        return {"overall_accuracy": self.correct / self.total,
                "instance_miou": self.iou_sum.sum() / self.shapes.sum(),
                "class_miou": float(per_class.mean()), "objects": int(self.shapes.sum()),
                "points": int(self.total),
                "per_category": {CATEGORIES[i][0]: {"miou": float(self.iou_sum[i] / self.shapes[i]),
                                                     "objects": int(self.shapes[i])}
                                 for i in range(16) if present[i]}}
