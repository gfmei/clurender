"""Batched OctFormer: curve keys, window attention and the encoder interface."""

import pytest
import torch

from models.octformer import Level, OctFormer, WindowAttention
from models.serialization import hilbert_key, morton_key


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def small(**overrides):
    config = dict(emb_dims=32, channels=(16, 32), blocks=(2, 2), heads=(2, 4), patch_size=8, dilation=2,
                  stride=4, k=8, voxel=0.25, fpn_channels=16, drop_path=0.0)
    config.update(overrides)
    return OctFormer(**config)


@pytest.mark.parametrize("key", [morton_key, hilbert_key])
def test_curve_keys_are_bijections_that_refine_coarser_orders(key):
    cells = torch.stack(torch.meshgrid(*[torch.arange(8)] * 3, indexing="ij"), -1).reshape(-1, 3)
    assert torch.equal(key(cells, 3).sort().values, torch.arange(512))
    fine = torch.randint(0, 64, (500, 3))
    assert torch.equal(key(fine, 6) >> 3, key(fine >> 1, 5))
    if key is hilbert_key:  # Consecutive Hilbert cells are face neighbors.
        order = cells[key(cells, 3).argsort()]
        assert (order[1:] - order[:-1]).abs().sum(-1).max() == 1


@pytest.mark.parametrize("dilation", [1, 2])
def test_window_attention_mixes_only_points_in_the_same_window(dilation):
    attention = WindowAttention(8, 2, patch_size=4, dilation=dilation).eval()
    level = Level(torch.rand(1, 16, 3), 0.25, "z", 4)
    x = torch.randn(1, 16, 8)
    changed = x.clone()
    changed[0, 5] += 1
    moved = (attention(changed, level) - attention(x, level)).abs().sum(-1)[0] > 1e-6
    expected = torch.zeros(16, dtype=torch.bool)
    if dilation == 1:
        expected[4:8] = True  # Point 5 is in the window of points 4..7.
    else:
        expected[[1, 3, 5, 7]] = True  # Block 0..7, window of odd offsets.
    assert torch.equal(moved, expected)


def test_full_attention_mixes_every_point_and_later_stages_use_it():
    attention = WindowAttention(8, 2, patch_size=0, dilation=1).eval()
    level = Level(torch.rand(1, 16, 3), 0.25, "z", 4)
    x = torch.randn(1, 16, 8)
    changed = x.clone()
    changed[0, 5] += 1
    assert ((attention(changed, level) - attention(x, level)).abs().sum(-1) > 1e-6).all()
    model = small(full_attention_from=1)
    windows = [[block.attention.patch_size for block in stage] for stage in model.stages]
    assert windows == [[8, 8], [0, 0]]
    assert [block.attention.dilation for block in model.stages[0]] == [1, 2]


@pytest.mark.parametrize("curve", ["z", "hilbert"])
def test_encoder_is_permutation_equivariant_translation_invariant_and_trainable(curve):
    model = small(curve=curve)
    points = torch.rand(2, 3, 96) * 2 - 1
    model.train()
    global_features, per_point, levels = model(points, return_levels=True)
    assert global_features.shape == (2, 32) and per_point.shape == (2, 32, 96)
    assert [level.shape for level in levels] == [(2, 16, 96)] * 2 and model.level_dims == (16, 16)
    (per_point.square().mean() + sum(level.mean() for level in levels)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        reference = model(points)[1]
        order = torch.randperm(96)
        torch.testing.assert_close(model(points[:, :, order])[1], reference[:, :, order], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(model(points + 0.3)[1], reference, atol=1e-4, rtol=1e-4)


def test_point_counts_that_do_not_fill_windows_or_pooling_groups():
    model = small().eval()
    with torch.no_grad():
        global_features, per_point = model(torch.rand(2, 3, 50) * 2 - 1)
    assert per_point.shape == (2, 32, 50) and torch.isfinite(per_point).all()
    with pytest.raises(ValueError, match="heads"):
        small(channels=(15, 32))


TINY = ["--octformer-channels", "16", "32", "--octformer-blocks", "1", "1", "--octformer-heads", "2", "4",
        "--octformer-patch-size", "8", "--octformer-k", "8", "--octformer-voxel", "0.25",
        "--octformer-fpn-channels", "16", "--emb-dims", "32"]


def test_pretrained_octformer_is_rebuilt_by_every_downstream_task(tmp_path):
    import json

    import numpy as np

    import main_finetune_partseg
    import main_partseg_probe
    from datasets.multiview import write_smoke_dataset
    from main_pretrain import parser, train
    from models.encoders import checkpoint_settings

    write_smoke_dataset(tmp_path / "data")
    train(parser().parse_args([
        "--root", str(tmp_path / "data"), "--output", str(tmp_path / "run"), "--model", "octformer", *TINY,
        "--device", "cpu", "--renderer", "torch", "--num-points", "32", "--num-views", "2", "--num-clusters", "4",
        "--image-size", "16", "--color-dims", "32", "24", "16", "16", "12", "8", "--batch-size", "2",
        "--points-per-pixel", "8", "--workers", "0", "--epochs", "1"]))
    backbone = tmp_path / "run" / "backbone.pth"
    assert checkpoint_settings(backbone)["model"] == "octformer"
    encoder, config, _ = main_partseg_probe.load_encoder(tmp_path / "run" / "last.pth")
    assert type(encoder).__name__ == "OctFormer" and config["serialization"] == "z"

    root = tmp_path / "parts"
    (root / "train_test_split").mkdir(parents=True)
    (root / "02691156").mkdir()
    for split, names in {"train": ["a", "b", "c"], "val": ["d"], "test": ["e", "f"]}.items():
        (root / "train_test_split" / f"shuffled_{split}_file_list.json").write_text(
            json.dumps([f"shape_data/02691156/{name}" for name in names]))
        for name in names:
            raw = np.zeros((40, 7), dtype=np.float32)
            raw[:, :3] = np.random.default_rng(len(name)).normal(size=(40, 3))
            raw[:, -1] = raw[:, 0] > 0
            np.savetxt(root / "02691156" / (name + ".txt"), raw)
    # The default --model dgcnn is overridden by the checkpoint.
    args = main_finetune_partseg.parser().parse_args([
        "--root", str(root), "--output", str(tmp_path / "seg"), "--pretrained", str(backbone),
        "--num-points", "32", "--epochs", "1", "--batch-size", "2", "--workers", "0", "--device", "cpu"])
    results = main_finetune_partseg.run(args)
    assert 0 <= results["last_epoch"]["instance_miou"] <= 1
    assert "loaded" not in results and json.loads((tmp_path / "seg" / "config.json").read_text())["model"] == "dgcnn"
