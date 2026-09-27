import json

import numpy as np
import pytest
import torch

from datasets.multiview import MultiViewPointCloudDataset
from prepare_shapenetpart import geometry_rgb, parser, prepare, select_sources


@pytest.fixture
def xyz_root(tmp_path):
    root = tmp_path / "PartAnnotation"
    splits = root / "train_test_split"
    splits.mkdir(parents=True)
    # No .seg files: preparation must succeed using only XYZ and split IDs.
    for split, ids in {"train": ["train", "test"], "val": ["val"], "test": ["test"]}.items():
        (splits / f"shuffled_{split}_file_list.json").write_text(json.dumps(
            [f"shape_data/02691156/{identity}" for identity in ids]))
    folder = root / "02691156/points"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for identity in ("train", "val", "test", "unlabeled"):
        points = rng.normal(size=(32, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= [.6, .4, 1.]
        np.savetxt(folder / (identity + ".pts"), points)
    return root


def test_selection_excludes_test_overlap_and_includes_unlabeled_geometry(xyz_root):
    sources, held_out = select_sources(xyz_root, "all-nontest")
    assert {name for name, _ in sources} == {"02691156/train", "02691156/val", "02691156/unlabeled"}
    assert held_out == ["02691156/test"]
    sources, _ = select_sources(xyz_root, "trainval")
    assert {name for name, _ in sources} == {"02691156/train", "02691156/val"}


def test_pca_shading_of_plane_and_normalized_coordinates(tmp_path):
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(16))) + [4, 7, 2]
    path = tmp_path / "plane.pts"
    np.savetxt(path, points)
    xyz, colors, digest = geometry_rgb(path, 8, 4)
    np.testing.assert_allclose(xyz.mean(0), 0, atol=1e-7)
    assert np.linalg.norm(xyz, axis=1).max() == pytest.approx(1)
    intensity = .25 + .55 * 2 / np.sqrt(6) + .20 * .5 / np.sqrt(5.25)
    np.testing.assert_allclose(colors, np.tile(intensity * np.array([.65, .75, .90]), (16, 1)), rtol=1e-6)
    assert len(digest) == 64


def test_small_clouds_are_deterministically_repeated_and_keep_calibration(xyz_root):
    path = xyz_root / "02691156/points/train.pts"
    xyz, colors, digest = geometry_rgb(path, 8, 40)
    again, repeated_colors, repeated_digest = geometry_rgb(path, 8, 40)
    assert xyz.shape == colors.shape == (40, 3)
    np.testing.assert_array_equal(xyz, again)
    np.testing.assert_array_equal(colors, repeated_colors)
    assert digest == repeated_digest
    assert all(np.any(np.all(xyz[:32] == point, axis=1)) for point in xyz[32:])
    np.testing.assert_allclose(xyz.mean(0), 0, atol=1e-7)
    assert np.linalg.norm(xyz, axis=1).max() == pytest.approx(1)


def test_preparation_is_calibrated_label_free_and_resumable(xyz_root, tmp_path):
    torch.set_num_threads(2)
    output = tmp_path / "paired"
    args = parser().parse_args(["--root", str(xyz_root), "--output", str(output),
                               "--device", "cpu", "--renderer", "torch", "--views", "3",
                               "--image-size", "16", "--radius", "1.5", "--sigma", "1",
                               "--minimum-points", "8", "--neighbors", "8", "--workers", "1"])
    result = prepare(args)
    assert result["objects"] == 3 and result["total_rgb_views"] == 9
    assert not result["part_labels_used"] and not result["failures"]
    data = MultiViewPointCloudDataset(output, 16, 3, 16, split=output / "train.txt")
    assert len(data) == 3 and data[0]["images"].shape == (3, 3, 16, 16)
    assert all("test.npz" not in str(path) for path in data.paths)
    for path in data.paths:
        with np.load(path) as archive:
            for image, transform, K in zip(archive["images"], archive["tsfms"], archive["K"]):
                # Independent pinhole projection checks saved RGB/camera alignment.
                camera = archive["points"] @ transform[:3, :3].T + transform[:3, 3]
                projection = camera @ K.T
                pixels = np.rint(projection[:, :2] / projection[:, 2:]).astype(int)
                visible = (camera[:, 2] > 0) & ((pixels >= 0) & (pixels < 16)).all(1)
                assert visible.any()
                assert (image[pixels[visible, 1], pixels[visible, 0]].max(-1) > 0).all()
    mtimes = [path.stat().st_mtime_ns for path in data.paths]
    assert prepare(args)["reused"] == 3
    assert mtimes == [path.stat().st_mtime_ns for path in data.paths]
    args.views = 4
    with pytest.raises(ValueError, match="different data/render settings"):
        prepare(args)


@pytest.mark.parametrize("points", [np.zeros((10, 3)), np.full((10, 3), np.nan), np.ones((2, 3))])
def test_bad_geometry_is_rejected(tmp_path, points):
    path = tmp_path / "bad.pts"
    np.savetxt(path, points)
    with pytest.raises(ValueError, match="Degenerate|Expected"):
        geometry_rgb(path, 8, 4)
