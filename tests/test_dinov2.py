"""DINOv2 feature extraction: crop geometry, PCA fitting and archive output, with a stub encoder."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import extract_dinov2
from datasets.multiview import crop_offset, feature_intrinsics
from extract_dinov2 import preprocess


class StubEncoder(torch.nn.Module):
    """Patch tokens from pooled colors, shaped like DINOv2's forward_features output."""

    def forward_features(self, x):
        pooled = F.adaptive_avg_pool2d(x, x.shape[-1] // 14)
        return {"x_norm_patchtokens": torch.cat((pooled, pooled.square()), 1).flatten(2).transpose(1, 2)}


def test_feature_intrinsics_follow_the_centered_crop_and_resize():
    K = torch.tensor([[221.7025, 0, 127.5], [0, 221.7025, 127.5], [0, 0, 1]], dtype=torch.float64)
    points = torch.rand(100, 3, dtype=torch.float64) * 2 - 1 + torch.tensor([0, 0, 2.7], dtype=torch.float64)
    pixels = points @ K.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    features = points @ feature_intrinsics(K, 256, 182, 32).T
    features = features[:, :2] / features[:, 2:]
    torch.testing.assert_close(features, (pixels - crop_offset(256, 182) + .5) / (182 / 32) - .5)
    # The image center maps to the center of the feature grid.
    center = feature_intrinsics(K, 256, 182, 32) @ torch.tensor([0, 0, 1.], dtype=torch.float64)
    torch.testing.assert_close(center[:2], torch.tensor([15.5, 15.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="margins"):
        crop_offset(256, 181)


def test_preprocess_crops_resizes_and_pools_foreground_to_patches():
    images = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[:, 1:8, 1:8] = True  # The top-left quarter of the 14-pixel crop.
    mask[:, 8:15, 14] = True  # A one-pixel-wide leg in the bottom-right patch.
    x, coverage = preprocess(images, mask, crop=14, input_size=28, grid=2)
    assert x.shape == (2, 3, 28, 28)
    torch.testing.assert_close(coverage, torch.tensor([[1, 0], [0, 1 / 7]]).expand(2, 2, 2))


def write_archives(root, count=3):
    rng = np.random.default_rng(0)
    entries = []
    for index in range(count):
        depths = np.zeros((2, 16, 16), dtype=np.float16)
        depths[:, 1:8, 1:8] = 2.5  # Exactly the top-left patch of the 14-pixel crop.
        images = (rng.random((2, 16, 16, 3)) * 255).astype(np.uint8) * (depths[..., None] > 0)
        path = root / "cat" / f"{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, images=images.astype(np.uint8), depths=depths)
        entries.append(f"cat/{index}.npz")
    return entries


def test_fit_and_extract_write_masked_projected_features_and_resume(tmp_path, monkeypatch):
    entries = write_archives(tmp_path / "paired")
    (tmp_path / "split.txt").write_text("\n".join(entries) + "\n")
    common = ["--root", str(tmp_path / "paired"), "--split", str(tmp_path / "split.txt"),
              "--output", str(tmp_path / "features"), "--input-size", "28", "--crop", "14",
              "--components", "3", "--workers", "0", "--device", "cpu"]
    for stage in ("fit", "extract"):
        args = extract_dinov2.parser().parse_args(["--stage", stage] + common)
        args.model_instance = StubEncoder()
        extract_dinov2.run(args)
    pca = torch.load(tmp_path / "features" / "pca.pt", weights_only=True)
    assert pca["components"].shape == (6, 3) and 0 < pca["explained"] <= 1 + 1e-6
    with np.load(tmp_path / "features" / "cat" / "0.npz") as archive:
        features, coverage = archive["features"], archive["coverage"]
    assert features.shape == (2, 3, 2, 2) and features.dtype == np.float16 and coverage.shape == (2, 2, 2)
    assert (coverage > 0).any() and (coverage == 0).any() and coverage.max() <= 1
    assert not features.transpose(0, 2, 3, 1)[coverage == 0].any()
    # A second run finds every archive done.
    stamp = (tmp_path / "features" / "cat" / "0.npz").stat().st_mtime_ns
    args = extract_dinov2.parser().parse_args(["--stage", "extract"] + common)
    args.model_instance = StubEncoder()
    extract_dinov2.run(args)
    assert (tmp_path / "features" / "cat" / "0.npz").stat().st_mtime_ns == stamp


def test_rendered_feature_error_is_normalized_and_ignores_splat_opacity():
    from models.losses import rendered_feature_error

    target, coverage = torch.randn(2, 4, 3, 3), torch.rand(2, 3, 3)
    alpha = torch.full((2, 1, 3, 3), .5)
    torch.testing.assert_close(rendered_feature_error(torch.cat((0 * target, alpha), 1), target, coverage),
                               torch.tensor(1.))
    # Features attenuated by sparse splats are exact once divided by their opacity.
    assert rendered_feature_error(torch.cat((alpha * target, alpha), 1), target, coverage) < 1e-6


def test_pretraining_renders_dinov2_features_end_to_end(tmp_path):
    import json

    from datasets.multiview import MultiViewPointCloudDataset, write_smoke_dataset
    from main_pretrain import parser as pretrain_parser, train

    data, features = tmp_path / "data", tmp_path / "features"
    write_smoke_dataset(data, num_samples=2, num_points=32, image_size=16, num_views=2)
    for stage in ("fit", "extract"):
        args = extract_dinov2.parser().parse_args([
            "--stage", stage, "--root", str(data), "--output", str(features), "--input-size", "28",
            "--crop", "16", "--components", "3", "--fit-coverage", ".25", "--workers", "0", "--device", "cpu"])
        args.model_instance = StubEncoder()
        extract_dinov2.run(args)
    item = MultiViewPointCloudDataset(data, 32, 2, 16, feature_root=features)[0]
    assert item["feature_maps"].shape == (2, 3, 2, 2) and item["feature_coverage"].shape == (2, 2, 2)
    torch.testing.assert_close(item["feature_K"], feature_intrinsics(item["K"], 16, 16, 2))
    train(pretrain_parser().parse_args([
        "--root", str(data), "--output", str(tmp_path / "run"), "--device", "cpu", "--renderer", "torch",
        "--emb-dims", "32", "--k", "4", "--num-points", "32", "--num-views", "2", "--num-clusters", "4",
        "--image-size", "16", "--batch-size", "2", "--workers", "0", "--epochs", "1",
        "--render-target", "dino", "--feature-root", str(features), "--color-decoder", "mlp",
        "--feature-points-per-pixel", "4"]))
    metrics = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[-1])
    # Four views, each starting from a zero prediction with normalized error 1.
    assert 0 < metrics["rendering"] <= 4 + 1e-3
