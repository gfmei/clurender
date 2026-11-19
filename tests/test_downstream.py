"""Downstream classification, part segmentation and SVM evaluation on tiny fixtures."""

import json
from argparse import Namespace
import pickle

import numpy as np
import pytest
import torch

from datasets.downstream import load_classification
import main_finetune_cls
import main_finetune_partseg
from models.dgcnn import DGCNN
from models.finetune import DGCNNClassifier, DGCNNPartSegmenter, load_encoder, subsample


@pytest.fixture(autouse=True)
def setup_torch():
    torch.set_num_threads(2)
    torch.manual_seed(3)


def clouds(count, points=64, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 5
    xyz = rng.normal(size=(count, points, 3)).astype(np.float32)
    xyz[..., 0] *= 1 + 2 * labels[:, None]  # Classes differ in shape, which normalization keeps.
    return xyz, labels


@pytest.fixture
def fewshot_root(tmp_path):
    root = tmp_path / "ModelNetFewshot"
    (root / "5way10shot").mkdir(parents=True)
    split = {}
    for name, count, seed in (("train", 10, 0), ("test", 10, 1)):
        xyz, labels = clouds(count, seed=seed)
        split[name] = [(np.concatenate((cloud, np.zeros_like(cloud)), 1), int(label), f"s{i}")
                       for i, (cloud, label) in enumerate(zip(xyz, labels))]
    with (root / "5way10shot" / "0.pkl").open("wb") as stream:
        pickle.dump(split, stream)
    return root


def small_encoder_file(tmp_path, full=False):
    encoder = DGCNN(32, 4, num_cls=-1)
    path = tmp_path / ("last.pth" if full else "backbone.pth")
    if full:
        torch.save({"args": {"model": "dgcnn"}, "model": {"backbone." + k: v for k, v in encoder.state_dict().items()}},
                   path)
    else:
        torch.save(encoder.state_dict(), path)
    return encoder, path


@pytest.mark.parametrize("full", [False, True])
def test_encoder_loads_strictly_from_backbone_or_full_checkpoint(tmp_path, full):
    encoder, path = small_encoder_file(tmp_path, full)
    model = DGCNNClassifier(40, emb_dims=32, k=4)
    assert load_encoder(model.encoder, path) == len(encoder.state_dict())
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(model.encoder.state_dict()[name], value)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_encoder(DGCNNClassifier(40, emb_dims=64, k=4).encoder, path)


def test_encoder_rejects_other_pretraining_backbones(tmp_path):
    torch.save({"args": {"model": "pointnet"}, "model": {}}, tmp_path / "last.pth")
    with pytest.raises(ValueError, match="pointnet"):
        load_encoder(DGCNN(32, 4, num_cls=-1), tmp_path / "last.pth")


def test_heads_and_multilevel_encoder_features_have_expected_shapes():
    points = torch.rand(2, 3, 48)
    classifier = DGCNNClassifier(15, emb_dims=32, k=4).eval()
    segmenter = DGCNNPartSegmenter(emb_dims=32, k=4).eval()
    assert classifier(points).shape == (2, 15)
    assert segmenter(points, torch.tensor([0, 15])).shape == (2, 50, 48)
    _, features, levels = segmenter.encoder(points, return_levels=True)
    assert [level.shape[1] for level in levels] == [64, 64, 128, 256] and features.shape == (2, 32, 48)
    # The DGCNN classification head expects max- and average-pooled features.
    assert DGCNN(32, 4, num_cls=40).eval()(points)[0].shape == (2, 40)


def test_subsampling_is_deterministic_for_testing_and_random_for_training():
    points = torch.rand(2, 3, 50)
    torch.testing.assert_close(subsample(points, 20), subsample(points, 20))
    sampled = subsample(points, 20, pool=30, generator=torch.Generator().manual_seed(0))
    assert sampled.shape == (2, 3, 20)
    # Every sampled point is one of the originals.
    distances = torch.cdist(sampled.transpose(1, 2), points.transpose(1, 2),
                            compute_mode="donot_use_mm_for_euclid_dist")
    assert (distances.amin(-1) < 1e-6).all()
    with pytest.raises(ValueError):
        subsample(points, 51)


def test_fewshot_loader_normalizes_and_accepts_both_folder_names(fewshot_root):
    points, labels, classes = load_classification("fewshot", fewshot_root, "train", way=5, shot=10, fold=0)
    assert points.shape == (10, 3, 64) and classes == 5 and labels.tolist() == [0, 1, 2, 3, 4] * 2
    assert torch.linalg.norm(points, dim=1).amax(-1).allclose(torch.ones(10))
    (fewshot_root / "5way10shot").rename(fewshot_root / "5way_10shot")
    assert len(load_classification("fewshot", fewshot_root, "test", way=5, shot=10, fold=0)[0]) == 10


def cls_args(root, output, **overrides):
    args = main_finetune_cls.parser().parse_args([
        "--dataset", "fewshot", "--root", str(root), "--output", str(output), "--num-points", "32",
        "--point-pool", "48", "--epochs", "2", "--batch-size", "4", "--k", "4", "--emb-dims", "32",
        "--votes", "2", "--device", "cpu"])
    return Namespace(**{**vars(args), **overrides})


def test_classification_finetuning_votes_resumes_and_reports_both_protocols(tmp_path, fewshot_root):
    _, path = small_encoder_file(tmp_path)
    results = main_finetune_cls.run(cls_args(fewshot_root, tmp_path / "run", pretrained=str(path)))
    assert set(results) >= {"last_epoch", "best_epoch_test_selected", "best_epoch_vote"}
    assert 0 <= results["best_epoch_vote"]["accuracy"] <= 1 and results["best_epoch_test_selected"]["epoch"] in (1, 2)
    resumed = main_finetune_cls.run(cls_args(fewshot_root, tmp_path / "run", epochs=3, resume=True))
    history = [json.loads(line)["epoch"] for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert history == [1, 2, 3] and resumed["epochs"] == 3


@pytest.fixture
def part_root(tmp_path):
    root = tmp_path / "shapenetpart"
    (root / "train_test_split").mkdir(parents=True)
    (root / "02691156").mkdir()
    for split, names in {"train": ["a", "b", "c"], "val": ["d"], "test": ["e", "f"]}.items():
        (root / "train_test_split" / f"shuffled_{split}_file_list.json").write_text(
            json.dumps([f"shape_data/02691156/{name}" for name in names]))
        for name in names:
            raw = np.zeros((40, 7), dtype=np.float32)
            raw[:, 0] = np.linspace(-1, 1, 40)
            raw[:, 1] = np.sin(np.arange(40)) * .1
            raw[:, -1] = raw[:, 0] > 0  # Airplane parts 0 and 1.
            np.savetxt(root / "02691156" / (name + ".txt"), raw)
    return root


def test_part_segmentation_finetuning_reports_last_and_best_mious(tmp_path, part_root):
    _, path = small_encoder_file(tmp_path)
    args = main_finetune_partseg.parser().parse_args([
        "--root", str(part_root), "--output", str(tmp_path / "seg"), "--pretrained", str(path),
        "--num-points", "32", "--epochs", "2", "--batch-size", "2", "--k", "4", "--emb-dims", "32",
        "--workers", "0", "--device", "cpu"])
    results = main_finetune_partseg.run(args)
    for key in ("last_epoch", "best_epoch_test_selected"):
        assert 0 <= results[key]["instance_miou"] <= 1 and results[key]["objects"] == 2
    assert json.loads((tmp_path / "seg/split_audit.json").read_text())["effective_counts"]["test"] == 2
    args.output, args.label_fraction = str(tmp_path / "few"), .5
    assert main_finetune_partseg.run(args)["label_fraction"] == .5
    assert json.loads((tmp_path / "few/split_audit.json").read_text())["labeled_objects"] == 2


def test_label_fraction_keeps_a_stratified_subset_repeated_with_new_samples(part_root):
    from datasets.shapenetpart import ShapeNetPartText, stratified_subset

    categories = np.repeat([0, 1, 2], [100, 10, 1])
    subset = stratified_subset(categories, .05, seed=0)
    assert np.bincount(categories[subset]).tolist() == [5, 1, 1]
    assert np.array_equal(subset, stratified_subset(categories, .05, seed=0))
    assert not np.array_equal(subset, stratified_subset(categories, .05, seed=1))
    full = ShapeNetPartText(part_root, "trainval", 32)
    half = ShapeNetPartText(part_root, "trainval", 32, label_fraction=.5)
    assert len(half.ids) == 2 and len(half) == 4 and half.fingerprint != full.fingerprint
    # The first repeat keeps the full-data sample; later repeats draw new points.
    np.testing.assert_array_equal(half[0]["points"], full[full.ids.index(half.ids[0])]["points"])
    assert not np.array_equal(half[0]["points"], half[2]["points"])
    with pytest.raises(ValueError, match="label_fraction"):
        ShapeNetPartText(part_root, "trainval", 32, label_fraction=0)


def test_svm_scores_frozen_features_on_modelnet40_files(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("sklearn")
    import main_svm

    xyz, labels = clouds(20)
    for split in ("train", "test"):
        with h5py.File(tmp_path / f"ply_data_{split}0.h5", "w") as archive:
            archive["data"], archive["label"] = xyz, labels[:, None].astype(np.uint8)
    _, path = small_encoder_file(tmp_path)
    args = main_svm.parser().parse_args(["--root", str(tmp_path), "--pretrained", str(path), "--num-points", "32",
                                         "--k", "4", "--emb-dims", "32", "--device", "cpu",
                                         "--output", str(tmp_path / "svm.json")])
    results = main_svm.run(args)
    assert json.loads((tmp_path / "svm.json").read_text()) == results and 0 <= results["accuracy"] <= 1
    args.pretrained = None  # Randomly initialized reference encoder.
    assert main_svm.run(args)["pretrained"] is None
    args.pretrained = str(path)
    # Features that reveal the class: the SVM fit and scoring must then be exact.
    monkeypatch.setattr(main_svm, "features", lambda encoder, points, args: np.eye(5)[labels])
    results = main_svm.run(args)
    assert results["accuracy"] == 1 and results["class_accuracy"] == 1


def test_frozen_encoder_epochs_keep_pretrained_weights_and_statistics(tmp_path, part_root):
    encoder, path = small_encoder_file(tmp_path)
    args = main_finetune_partseg.parser().parse_args([
        "--root", str(part_root), "--output", str(tmp_path / "seg"), "--pretrained", str(path),
        "--num-points", "32", "--epochs", "2", "--batch-size", "2", "--k", "4", "--emb-dims", "32",
        "--workers", "0", "--device", "cpu", "--freeze-encoder-epochs", "2", "--encoder-lr-scale", "0.1"])
    main_finetune_partseg.run(args)
    tuned = torch.load(tmp_path / "seg" / "last.pth", weights_only=True)["model"]
    # Weights and BatchNorm running statistics of the encoder stay exactly as pretrained.
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(tuned["encoder." + name], value, rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path / "seg" / "metrics.jsonl").read_text().splitlines()]
    assert [r["encoder_frozen"] for r in records] == [True, True]
    model = DGCNNPartSegmenter(emb_dims=32, k=4)
    optimizer, _ = main_finetune_partseg.build_optimizer(model, args)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([args.lr, args.lr * 0.1])
    assert len(optimizer.param_groups[1]["params"]) == len(list(model.encoder.parameters()))
    with pytest.raises(ValueError, match="freeze-encoder-epochs"):
        args.freeze_encoder_epochs = 3
        main_finetune_partseg.run(args)
