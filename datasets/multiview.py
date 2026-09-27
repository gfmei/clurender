"""Paired point clouds, RGB views and calibrated OpenCV cameras.

Each .npz archive has ``points`` (N,3), ``images`` (V,H,W,3), ``tsfms``
(V,4,4), and ``K`` (V,3,3) or (3,3). Images are uint8 or floats in [0,1].
Points and cameras must already share the same world coordinate frame.
Archives from prepare_shapenet.py also hold ``normals`` (N,3), ``colors``
(N,3), ``visibility`` (V,N) and ``depths`` (V,H,W); training ignores them.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def project_points(points, tsfms, K):
    """Project world points into calibrated views, e.g. to lift image features.

    ``points`` (..., N, 3), ``tsfms`` (..., V, 4, 4) world-to-camera and ``K``
    (..., V, 3, 3) follow the archive conventions. Returns pixel coordinates
    (..., V, N, 2), with (0, 0) at the center of the top-left pixel, and camera
    depths (..., V, N). Coordinates are only meaningful where depth > 0.
    """
    camera = points.unsqueeze(-3) @ tsfms[..., :3, :3].transpose(-1, -2) + tsfms[..., None, :3, 3]
    depth = camera[..., 2]
    return (camera @ K.transpose(-1, -2))[..., :2] / depth.unsqueeze(-1), depth


def sample_points(points, count, rng, method="fps"):
    total = len(points)
    if total < count:
        raise ValueError(f"Requested {count} points, but sample contains only {total}")
    if total == count:
        return points.copy()
    if method == "random":
        return points[rng.choice(total, count, replace=False)]
    indices = np.empty(count, dtype=np.int64)
    distance = np.full(total, np.inf)
    farthest = ((points - points.mean(0)) ** 2).sum(1).argmax()
    for i in range(count):
        indices[i] = farthest
        distance = np.minimum(distance, ((points - points[farthest]) ** 2).sum(1))
        # Exclude chosen indices even for coincident points.
        distance[indices[:i + 1]] = -1
        farthest = distance.argmax()
    return points[indices]


class MultiViewPointCloudDataset(Dataset):
    def __init__(self, root, num_points=1024, num_views=8, image_size=256,
                 split=None, sampling="fps", seed=0):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Paired dataset directory does not exist: {root}")
        if split is None:
            self.paths = sorted(self.root.rglob("*.npz"))
        else:
            entries = Path(split).read_text().splitlines()
            self.paths = [self.root / line.strip() for line in entries
                          if line.strip() and not line.lstrip().startswith("#")]
        if not self.paths:
            raise ValueError(f"No paired .npz samples found in {root}")
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing sample in split: {path}")
        if num_points < 1 or num_views < 1 or image_size < 1:
            raise ValueError("Point count, view count, and image size must be positive")
        if sampling not in ("fps", "random"):
            raise ValueError("sampling must be fps or random")
        self.num_points, self.num_views = num_points, num_views
        self.image_size, self.sampling = image_size, sampling
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        with np.load(path, allow_pickle=False) as archive:
            required = {"points", "images", "tsfms", "K"}
            if not required.issubset(archive.files):
                raise ValueError(f"{path}: missing arrays {sorted(required - set(archive.files))}")
            points = np.asarray(archive["points"], dtype=np.float32)
            images = archive["images"]
            transforms = np.asarray(archive["tsfms"], dtype=np.float32)
            intrinsics = np.asarray(archive["K"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{path}: points must have shape (N, 3)")
        if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[:3]) < 1:
            raise ValueError(f"{path}: images must have shape (V, H, W, 3)")
        views, height, width, _ = images.shape
        if transforms.shape != (views, 4, 4):
            raise ValueError(f"{path}: tsfms must have shape (V, 4, 4)")
        if intrinsics.shape == (3, 3):
            intrinsics = np.broadcast_to(intrinsics, (views, 3, 3)).copy()
        if intrinsics.shape != (views, 3, 3):
            raise ValueError(f"{path}: K must have shape (3, 3) or (V, 3, 3)")
        if not all(np.isfinite(array).all() for array in (points, images, transforms, intrinsics)):
            raise ValueError(f"{path}: non-finite data or calibration")
        if not np.allclose(transforms[:, 3], [0, 0, 0, 1]):
            raise ValueError(f"{path}: tsfms must be homogeneous world-to-camera transforms")
        rotations = transforms[:, :3, :3]
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
            raise ValueError(f"{path}: camera rotations must be proper orthonormal matrices")
        if (not np.allclose(intrinsics[:, 2], [0, 0, 1])
                or np.any(intrinsics[:, (0, 1), (0, 1)] <= 0)):
            raise ValueError(f"{path}: invalid pinhole intrinsics")
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255
        elif np.issubdtype(images.dtype, np.floating) and images.min() >= 0 and images.max() <= 1:
            images = images.astype(np.float32)
        else:
            raise ValueError(f"{path}: images must be uint8 or floating RGB in [0, 1]")
        if views < self.num_views:
            raise ValueError(f"{path}: requested {self.num_views} views but only {views} are available")
        selected = rng.choice(views, self.num_views, replace=False)
        points = sample_points(points, self.num_points, rng, self.sampling)
        images = torch.from_numpy(images[selected]).permute(0, 3, 1, 2).contiguous()
        intrinsics = torch.from_numpy(intrinsics[selected].copy())
        if (height, width) != (self.image_size, self.image_size):
            images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear",
                                   align_corners=False, antialias=True)
            # Account for the half-pixel convention of align_corners=False.
            for axis, old_size in enumerate((width, height)):
                scale = self.image_size / old_size
                intrinsics[:, axis, :] *= scale
                intrinsics[:, axis, 2] += (scale - 1) / 2
        return {"points": torch.from_numpy(points.copy()).T.contiguous(),
                "images": images, "tsfms": torch.from_numpy(transforms[selected].copy()),
                "K": intrinsics, "sample_id": str(path.relative_to(self.root))}


def orbit_cameras(num_views, image_size, distance=2.7, elevation=15.0):
    """Deterministic OpenCV cameras looking at the origin, with Y world-up."""
    transforms = []
    elevation = np.deg2rad(elevation)
    for angle in np.linspace(0, 2 * np.pi, num_views, endpoint=False):
        center = distance * np.array([np.cos(elevation) * np.sin(angle),
                                      np.sin(elevation), np.cos(elevation) * np.cos(angle)])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0., 1., 0.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ center
        transforms.append(transform)
    focal = image_size / (2 * np.tan(np.deg2rad(60) / 2))
    K = np.array([[focal, 0, (image_size - 1) / 2], [0, focal, (image_size - 1) / 2],
                  [0, 0, 1]], dtype=np.float32)
    return np.stack(transforms), np.broadcast_to(K, (num_views, 3, 3)).copy()


def write_smoke_dataset(root, num_samples=4, num_points=32, image_size=16, num_views=2):
    """Write a tiny colored-sphere fixture. This is not research training data."""
    from models.common import points_to_ndc, transform_points_tsfm
    from models.renderer import PointRenderer, RenderConfig

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    transforms, intrinsics = orbit_cameras(num_views, image_size)
    renderer = PointRenderer(RenderConfig(render_size=image_size, backend="torch",
                                          radius=2, sigma=1, points_per_pixel=8))
    rng = np.random.default_rng(42)
    with torch.no_grad():
        for index in range(num_samples):
            points = rng.normal(size=(num_points, 3)).astype(np.float32)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            points *= rng.uniform(0.6, 0.9)
            tensor = torch.from_numpy(points).unsqueeze(0)
            colors = (tensor + 1) / 2
            images = []
            for transform, K in zip(transforms, intrinsics):
                camera = transform_points_tsfm(tensor, torch.from_numpy(transform).unsqueeze(0))
                ndc = points_to_ndc(camera, torch.from_numpy(K).unsqueeze(0), [image_size, image_size])
                image = renderer(ndc, colors)["feats"][0].permute(1, 2, 0)
                images.append((image.clamp(0, 1).numpy() * 255).round().astype(np.uint8))
            np.savez_compressed(root / f"sample_{index:03d}.npz", points=points,
                                images=np.stack(images), tsfms=transforms, K=intrinsics)
