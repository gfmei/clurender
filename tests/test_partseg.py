import json
import shutil

import numpy as np
import pytest
import torch

from datasets.shapenetpart import PartSegMetrics, ShapeNetPartText, split_ids
from main_partseg_probe import category_mask, load_encoder, parser, predict_parts, run
from models.dgcnn import DGCNN


@pytest.fixture
def part_root(tmp_path):
    root = tmp_path / "data"
    (root / "train_test_split").mkdir(parents=True)
    (root / "02691156").mkdir()
    for split, names in {"train": ["a", "b"], "val": ["c"], "test": ["d"]}.items():
        (root / "train_test_split" / f"shuffled_{split}_file_list.json").write_text(
            json.dumps([f"shape_data/02691156/{name}" for name in names]))
        for name in names:
            raw = np.zeros((16, 7), dtype=np.float32)
            raw[:, 0] = np.linspace(-1, 1, 16)
            raw[:, 1] = np.sin(np.arange(16)) * .1
            raw[:, -1] = raw[:, 0] > 0
            np.savetxt(root / "02691156" / (name + ".txt"), raw)
    return root


def test_part_sampling_preserves_labels_is_repeatable_and_disjoint(part_root):
    train = ShapeNetPartText(part_root, "trainval", 32, seed=5)
    test = ShapeNetPartText(part_root, "test", 32, seed=5)
    a, b = train[0], train[0]
    assert len(train) == 3 and len(test) == 1
    assert not set(train.ids) & set(test.ids)
    np.testing.assert_array_equal(a["points"], b["points"])
    np.testing.assert_array_equal(a["parts"], a["points"][0] > 0)
    assert np.linalg.norm(a["points"], axis=0).max() <= 1.00001
    assert a["points"].shape == (3, 32)


def test_overlapping_splits_are_rejected(part_root):
    split = part_root / "train_test_split/shuffled_test_file_list.json"
    split.write_text(json.dumps(["shape_data/02691156/a"]))
    with pytest.raises(ValueError, match="overlap"):
        split_ids(part_root)


def test_invalid_part_category_pair_is_rejected(part_root):
    path = part_root / "02691156/a.txt"
    raw = np.loadtxt(path)
    raw[0, -1] = 49
    np.savetxt(path, raw)
    with pytest.raises(ValueError, match="Part labels disagree"):
        ShapeNetPartText(part_root)[0]


def make_original_format(part_root):
    root = part_root.parent / "PartAnnotation"
    shutil.copytree(part_root / "train_test_split", root / "train_test_split")
    for path in part_root.glob("*/*.txt"):
        data = np.loadtxt(path)
        points = root / path.parent.name / "points" / (path.stem + ".pts")
        labels = root / path.parent.name / "expert_verified" / "points_label" / (path.stem + ".seg")
        points.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(points, data[:, :3])
        np.savetxt(labels, data[:, -1] + 1)
    return root


@pytest.mark.parametrize("synset,offset", [("02691156", 0), ("02773838", 4), ("03790512", 30)])
def test_original_format_matches_normal_text_sampling_and_global_labels(part_root, synset, offset):
    root = make_original_format(part_root)
    for directory in (root, part_root):
        if synset != "02691156":
            (directory / "02691156").rename(directory / synset)
        for path in (directory / "train_test_split").glob("*.json"):
            path.write_text(path.read_text().replace("02691156", synset))
    for path in (part_root / synset).glob("*.txt"):
        data = np.loadtxt(path)
        data[:, -1] += offset
        np.savetxt(path, data)
    original = ShapeNetPartText(root, num_points=32, seed=5)
    normals = ShapeNetPartText(part_root, num_points=32, seed=5)
    for i in range(len(original)):
        for field in ("points", "parts", "category"):
            np.testing.assert_array_equal(original[i][field], normals[i][field])
    path = original.label_paths[0]
    path.write_text(path.read_text() + "\n")
    assert ShapeNetPartText(root, num_points=32, seed=5).fingerprint != original.fingerprint


@pytest.mark.parametrize("bad_labels", ["1\n", "0\n" * 16, "1.5\n" * 16, "nan\n" * 16])
def test_original_format_rejects_invalid_or_misaligned_labels(part_root, bad_labels):
    data = ShapeNetPartText(make_original_format(part_root))
    data.label_paths[0].write_text(bad_labels)
    with pytest.raises(ValueError, match="count mismatch|Part labels"):
        data[0]


def test_excluding_overlap_preserves_test_and_deduplicates_fitting(part_root):
    for split, names in {"test": ["a", "d"], "val": ["b", "c"]}.items():
        (part_root / "train_test_split" / f"shuffled_{split}_file_list.json").write_text(
            json.dumps(["shape_data/02691156/" + name for name in names]))
    train = ShapeNetPartText(part_root, "trainval", exclude_overlap=True)
    test = ShapeNetPartText(part_root, "test", exclude_overlap=True)
    assert train.ids == ["02691156/b", "02691156/c"]
    assert test.ids == ["02691156/a", "02691156/d"]
    assert train.split_audit["train_test_overlap"] == ["02691156/a"]
    assert train.split_audit["train_val_overlap"] == ["02691156/b"]


def test_iou_absent_parts_class_weighting_and_singletons():
    meter = PartSegMetrics()
    meter.update([[0, 0]], [[0, 1]], [0])
    meter.update([[0, 1], [4, 5]], [[0, 1], [4, 5]], [0, 1])
    result = meter.compute()
    assert result["overall_accuracy"] == pytest.approx(5 / 6)
    assert result["instance_miou"] == pytest.approx(.875)
    assert result["class_miou"] == pytest.approx(.90625)
    assert result["objects"] == 3


def test_category_constrained_prediction_excludes_other_object_parts():
    logits = torch.zeros(2, 3, 50)
    logits[:, :, 49] = 100
    logits[0, :, 2] = 1
    logits[1, :, 5] = 1
    pred = predict_parts(logits, torch.tensor([0, 1]), category_mask())
    torch.testing.assert_close(pred, torch.tensor([[2, 2, 2], [5, 5, 5]]))


def make_checkpoint(path):
    torch.manual_seed(1)
    model = DGCNN(8, 2)
    torch.save({"args": {"model": "dgcnn", "emb_dims": 8, "k": 2,
                          "root": "synthetic-test", "epochs": 1},
                "model": {"backbone." + key: value for key, value in model.state_dict().items()}}, path)
    return model


def test_encoder_transfer_is_complete_frozen_and_strict(tmp_path):
    path = tmp_path / "encoder.pth"
    source = make_checkpoint(path)
    encoder, _, weights = load_encoder(path)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(encoder.state_dict()[name], value, rtol=0, atol=0)
    broken = torch.load(path, weights_only=True)
    del broken["model"]["backbone.conv1.0.weight"]
    torch.save(broken, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder(path)


def test_linear_probe_trains_resumes_and_tests_held_out_objects(part_root, tmp_path):
    torch.set_num_threads(2)
    pretrained = tmp_path / "encoder.pth"
    source = make_checkpoint(pretrained)
    common = ["--root", str(part_root), "--pretrained", str(pretrained),
              "--cache", str(tmp_path / "cache"), "--device", "cpu", "--num-points", "8",
              "--batch-size", "2", "--extract-batch-size", "2", "--workers", "0"]
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resume"
    run(parser().parse_args(common + ["--output", str(uninterrupted), "--epochs", "2"]))
    run(parser().parse_args(common + ["--output", str(resumed), "--epochs", "1"]))
    result = run(parser().parse_args(common + ["--output", str(resumed), "--epochs", "2",
                                             "--resume", str(resumed / "last.pth")]))
    actual = torch.load(resumed / "last.pth", weights_only=True)
    expected = torch.load(uninterrupted / "last.pth", weights_only=True)
    assert actual["epoch"] == 2
    for key in actual["head"]:
        torch.testing.assert_close(actual["head"][key], expected["head"][key], rtol=0, atol=0)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(actual["encoder"][key], value, rtol=0, atol=0)
    assert result["objects"] == 1 and result["train_objects"] == 3
    assert result["points"] == 8 and 0 <= result["instance_miou"] <= 1
    # Re-evaluate an already completed checkpoint without entering the epoch loop.
    run(parser().parse_args(common + ["--output", str(resumed), "--epochs", "2",
                                      "--resume", str(resumed / "last.pth")]))
