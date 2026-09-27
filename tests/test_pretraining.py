import importlib.util
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from datasets.multiview import MultiViewPointCloudDataset, orbit_cameras, write_smoke_dataset
from main_pretrain import parser, train
from models.clurender import CluRender, ClusterNet, ot_assign
from models.color import UNetTransformer
from models.common import points_to_ndc, sinkhorn, transform_points_tsfm
from models.dgcnn import DGCNN
from models.losses import BalancedClustering, ImageWassersteinLoss, SlicedWassersteinDistance
from models.pointnet import PointNet
from models.renderer import PointRenderer, RenderConfig


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.set_num_threads(2)
    torch.manual_seed(8)


def test_rigid_transform_inverse_and_known_projection():
    points = torch.tensor([[[1., 0., 2.], [0., 1., 3.]]])
    transform = torch.tensor([[[0., -1., 0., 1.], [1., 0., 0., 2.],
                               [0., 0., 1., 3.], [0., 0., 0., 1.]]])
    camera = transform_points_tsfm(points, transform)
    torch.testing.assert_close(camera, torch.tensor([[[1., 3., 5.], [0., 2., 6.]]]))
    torch.testing.assert_close(transform_points_tsfm(camera, transform, inverse=True), points)
    K = torch.tensor([[[2., 0., 2.5], [0., 2., 1.5], [0., 0., 1.]]])
    ndc = points_to_ndc(torch.tensor([[[0., 0., 2.], [1., 1., 2.], [0., 0., -2.]]]), K, [4, 6])
    torch.testing.assert_close(ndc, torch.tensor([[[0., 0., 2.], [-.5, -.5, 2.], [0., 0., -2.]]]))


@pytest.mark.parametrize("shape", [(1, 5, 3), (2, 5, 1), (1, 1, 1)])
def test_sinkhorn_preserves_singleton_axes_and_marginals(shape):
    batch, count, clusters = shape
    cost = torch.rand(shape, dtype=torch.float64)
    p = torch.full((batch, count), 1 / count, dtype=cost.dtype)
    q = torch.full((batch, clusters), 1 / clusters, dtype=cost.dtype)
    plan, _ = sinkhorn(cost, p, q, epsilon=.1, max_iter=500, thresh=1e-8)
    assert plan.shape == shape
    torch.testing.assert_close(plan.sum(-1), p, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(plan.sum(-2), q, atol=1e-7, rtol=1e-7)


def test_transport_prefers_nearest_prototype_and_legacy_singleton():
    cost = torch.tensor([[[0., 1.], [1., 0.]]])
    marginal = torch.full((1, 2), .5)
    plan, _ = sinkhorn(cost, marginal, marginal, epsilon=.01)
    assert plan[0, 0, 0] > .49 and plan[0, 1, 0] < 1e-5
    plan, _ = ot_assign(torch.zeros(1, 3, 1), torch.zeros(1, 3, 1), dst="eu")
    assert plan.shape == (1, 1, 1)
    torch.testing.assert_close(plan, torch.ones_like(plan))


def test_clustering_detaches_labels_but_trains_prototypes_and_encoder():
    cluster = BalancedClustering(16, 3, epsilon=.1, iterations=200)
    features = torch.randn(1, 16, 12, requires_grad=True)
    loss, details = cluster(features, torch.randn(1, 3, 12), return_details=True)
    assert not details["labels"].requires_grad
    assert details["prototypes"].requires_grad
    torch.testing.assert_close(details["labels"].sum(-1), torch.ones(1, 12), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(details["labels"].sum(1), torch.full((1, 3), 4.), atol=1e-4, rtol=1e-4)
    loss.backward()
    assert features.grad.norm() > 0
    assert cluster.head[-1].weight.grad.norm() > 0


@pytest.mark.parametrize("count", [1, 2, 19, 32])
def test_color_decoder_restores_resolution_and_uses_all_stages(count):
    model = UNetTransformer(16, d_dims=(16, 12, 8), u_dims=(8, 8, 4))
    features = torch.randn(2, count, 16, requires_grad=True)
    colors = model(torch.randn(2, count, 3), features)
    assert colors.shape == (2, 3, count)
    colors.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.norm() > 0
    for stage in list(model.encoder_layers) + list(model.decoder_layers):
        assert stage.value.weight.grad.norm() > 0


def test_color_decoder_does_not_attend_across_batches():
    model = UNetTransformer(16, d_dims=(16, 12, 8), u_dims=(8, 8, 4)).eval()
    points, features = torch.randn(2, 19, 3), torch.randn(2, 19, 16)
    together = model(points, features)
    separate = torch.cat([model(points[i:i + 1], features[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-5)


def test_renderer_occlusion_empty_pixels_and_color_gradients():
    renderer = PointRenderer(RenderConfig(render_size=1, backend="torch", points_per_pixel=2))
    points = torch.tensor([[[0., 0., 1.], [0., 0., 2.], [0., 0., -1.]]])
    colors = torch.tensor([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]], requires_grad=True)
    result = renderer(points, colors)
    torch.testing.assert_close(result["feats"][0, :, 0, 0], torch.tensor([.99, 0, .0099]))
    result["feats"].sum().backward()
    assert colors.grad[0, 0].norm() > colors.grad[0, 1].norm() > 0
    assert colors.grad[0, 2].norm() == 0
    empty = renderer(points[:, 2:], colors[:, 2:])
    assert torch.equal(empty["feats"], torch.zeros_like(empty["feats"]))
    assert empty["depth"].item() == 0 and empty["mask"].item() == 0


def test_renderer_rectangular_pixel_location_and_position_gradient():
    renderer = PointRenderer(RenderConfig(render_size=(3, 5), radius=.8, sigma=.5,
                                          backend="torch", points_per_pixel=1))
    K = torch.tensor([[[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]]])
    camera = torch.tensor([[[-.9, 0., 1.]]], requires_grad=True)
    ndc = points_to_ndc(camera, K, [3, 5])
    image = renderer(ndc, torch.ones(1, 1, 3))["feats"]
    assert image[0, 0].argmax().item() == 6  # row=1, column=1
    image.sum().backward()
    assert torch.isfinite(camera.grad).all() and camera.grad.norm() > 0


def test_image_ot_matches_single_pixel_analytic_cost():
    loss = ImageWassersteinLoss(backend="tensorized")
    rendered = torch.tensor([[[[1.]], [[.5]], [[0.]]]], requires_grad=True)
    value = loss(rendered, torch.zeros_like(rendered))
    torch.testing.assert_close(value, torch.tensor(.625), atol=1e-5, rtol=1e-5)
    value.backward()
    torch.testing.assert_close(rendered.grad, rendered.detach())


def test_sliced_identical_images_have_finite_zero_gradient():
    images = torch.rand(1, 3, 5, 3, requires_grad=True)
    loss = SlicedWassersteinDistance(17)(images, images.detach())
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(images.grad).all() and images.grad.norm() == 0


def test_image_ot_identical_single_pixel_is_well_defined():
    image = torch.zeros(1, 3, 1, 1, requires_grad=True)
    loss = ImageWassersteinLoss(backend="tensorized")(image, image.detach())
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize("backbone", ["dgcnn", "pointnet"])
def test_rendering_loss_alone_reaches_backbone_and_color_decoder(backbone):
    net = DGCNN(32, 4) if backbone == "dgcnn" else PointNet(32, feature_transform=True)
    model = CluRender(net, 32, 4, RenderConfig(render_size=8, backend="torch"),
                      color_dims=(32, 24, 16, 16, 12, 8))
    points = torch.rand(2, 3, 32) - .5
    transforms, intrinsics = orbit_cameras(2, 8)
    transforms = torch.from_numpy(transforms).unsqueeze(0).repeat(2, 1, 1, 1)
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).repeat(2, 1, 1, 1)
    images = torch.rand(2, 2, 3, 8, 8)
    losses = model(points, images, transforms, intrinsics, return_details=True)
    torch.testing.assert_close(losses["loss"], losses["clustering"] + losses["rendering"] + losses["transform"])
    losses["rendering"].backward()
    assert model.color.final_conv.weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in net.parameters())
    assert all(p.grad is None for p in model.cluster.parameters())
    model.eval()
    with torch.no_grad():
        tensor_loss = model(points, images, transforms, intrinsics)
        list_loss = model(points, list(images.unbind(1)), list(transforms.unbind(1)), list(intrinsics.unbind(1)))
        torch.testing.assert_close(tensor_loss, list_loss)
        assert model(points, return_embedding=True).shape == (2, 32)


def test_legacy_clustering_remains_usable_without_renderer_dependencies():
    model = ClusterNet(DGCNN(32, 4), dim=32, num_clus=4)
    loss, regularizer = model(torch.rand(2, 3, 16))
    (loss + regularizer).backward()
    assert loss.isfinite() and regularizer.device == loss.device


def test_default_paper_widths_complete_a_training_step():
    model = CluRender(DGCNN(1024, 20), render_cfg=RenderConfig(render_size=8, backend="torch"))
    points = torch.rand(2, 3, 64) - .5
    transforms, K = orbit_cameras(1, 8)
    losses = model(points, torch.rand(2, 1, 3, 8, 8),
                   torch.from_numpy(transforms).unsqueeze(0).repeat(2, 1, 1, 1),
                   torch.from_numpy(K).unsqueeze(0).repeat(2, 1, 1, 1), return_details=True)
    losses["loss"].backward()
    assert torch.isfinite(model.backbone.conv5[0].weight.grad).all()
    assert model.color.final_conv.weight.grad.norm() > 0


def test_sliced_fitting_completes_a_joint_training_step():
    model = CluRender(DGCNN(32, 4), 32, 4, RenderConfig(render_size=8, backend="torch"),
                      image_loss="sliced", color_dims=(32, 24, 16, 16, 12, 8))
    transforms, K = orbit_cameras(1, 8)
    loss = model(torch.rand(2, 3, 16) - .5, torch.rand(2, 1, 3, 8, 8),
                 torch.from_numpy(transforms).unsqueeze(0).repeat(2, 1, 1, 1),
                 torch.from_numpy(K).unsqueeze(0).repeat(2, 1, 1, 1))
    loss.backward()
    assert torch.isfinite(loss) and model.color.final_conv.weight.grad.norm() > 0


def test_dataset_pairs_views_and_rescales_intrinsics(tmp_path):
    points = np.arange(48, dtype=np.float32).reshape(16, 3)
    transforms, K = orbit_cameras(3, 4)
    images = np.stack([np.full((4, 4, 3), i * 50, dtype=np.uint8) for i in range(3)])
    np.savez(tmp_path / "one.npz", points=points, images=images, tsfms=transforms, K=K)
    dataset = MultiViewPointCloudDataset(tmp_path, 16, 3, 8)
    sample = dataset[0]
    torch.testing.assert_close(sample["points"].T, torch.from_numpy(points))
    for view in range(3):
        index = round(sample["images"][view, 0, 0, 0].item() * 255 / 50)
        torch.testing.assert_close(sample["tsfms"][view], torch.from_numpy(transforms[index]))
        assert sample["K"][view, 0, 2] == 3.5
        assert sample["K"][view, 0, 0] == 2 * K[index, 0, 0]
    torch.testing.assert_close(sample["images"], dataset[0]["images"])
    dataset.set_epoch(7)
    torch.testing.assert_close(dataset[0]["points"], sample["points"])


def test_dataset_rejects_missing_calibration_and_inadequate_views(tmp_path):
    np.savez(tmp_path / "bad.npz", points=np.zeros((10, 3)))
    dataset = MultiViewPointCloudDataset(tmp_path, 8, 2, 8)
    with pytest.raises(ValueError, match="missing arrays"):
        dataset[0]
    (tmp_path / "bad.npz").unlink()
    write_smoke_dataset(tmp_path, num_samples=1, num_points=8, num_views=1, image_size=8)
    dataset = MultiViewPointCloudDataset(tmp_path, 8, 2, 8)
    with pytest.raises(ValueError, match="only 1"):
        dataset[0]


def test_shapenet_dataset_uses_paired_loader_without_legacy_dependencies(tmp_path):
    from datasets.shapenet.shapenet import ShapeNetDataset
    write_smoke_dataset(tmp_path, num_samples=1, num_points=8, num_views=1, image_size=8)
    sample = ShapeNetDataset(tmp_path, num_points=8, num_views=1, image_size=8)[0]
    assert sample["points"].shape == (3, 8) and sample["images"].shape == (1, 3, 8, 8)


def test_resume_exactly_matches_uninterrupted_training(tmp_path):
    data = tmp_path / "data"
    write_smoke_dataset(data, num_points=32)
    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    common = ["--smoke", "--root", str(data), "--threads", "2"]
    train(parser().parse_args(common + ["--output", str(complete), "--epochs", "2"]))
    train(parser().parse_args(common + ["--output", str(resumed), "--epochs", "1"]))
    train(parser().parse_args(["--resume", str(resumed / "last.pth"), "--epochs", "2",
                              "--device", "cpu", "--workers", "0", "--threads", "2"]))
    expected = torch.load(complete / "last.pth", weights_only=True)
    actual = torch.load(resumed / "last.pth", weights_only=True)
    assert actual["epoch"] == 2 and actual["global_step"] == 4
    assert actual["scheduler"] == expected["scheduler"]
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    backbone = DGCNN(32, 4)
    backbone.load_state_dict(torch.load(resumed / "backbone.pth", weights_only=True))


@pytest.mark.skipif(importlib.util.find_spec("pytorch3d") is None, reason="PyTorch3D unavailable")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_pytorch3d_renderer_matches_reference(device):
    config = dict(render_size=(5, 7), radius=1.5, sigma=1, points_per_pixel=4)
    points = torch.rand(2, 12, 3, device=device) * 2 - 1
    points[..., 2] += 1.2
    points.requires_grad_()
    features = torch.rand(2, 12, 3, device=device, requires_grad=True)
    reference = PointRenderer(RenderConfig(**config, backend="torch"))(points, features)
    accelerated = PointRenderer(RenderConfig(**config, backend="pytorch3d"))(points, features)
    torch.testing.assert_close(reference["feats"], accelerated["feats"], atol=2e-5, rtol=2e-5)
    expected_grad = torch.autograd.grad(reference["feats"].square().sum(), (points, features))
    actual_grad = torch.autograd.grad(accelerated["feats"].square().sum(), (points, features))
    for expected, actual in zip(expected_grad, actual_grad):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(importlib.util.find_spec("pykeops") is None, reason="KeOps unavailable")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_online_ot_matches_dense_loss_and_gradient(device):
    rendered = torch.rand(1, 3, 4, 4, device=device, requires_grad=True)
    target = torch.rand_like(rendered)
    dense = ImageWassersteinLoss(backend="tensorized")(rendered, target)
    dense_gradient, = torch.autograd.grad(dense, rendered)
    online = ImageWassersteinLoss(backend="online")(rendered, target)
    online_gradient, = torch.autograd.grad(online, rendered)
    torch.testing.assert_close(online, dense, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(online_gradient, dense_gradient, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(importlib.util.find_spec("pytorch3d") is None, reason="PyTorch3D unavailable")
def test_textured_mesh_preparation_normalizes_and_preserves_calibration(tmp_path):
    from PIL import Image
    from prepare_shapenet import prepare

    root, output = tmp_path / "meshes", tmp_path / "paired"
    mesh = root / "03001627" / "fixture" / "models"
    mesh.mkdir(parents=True)
    (mesh / "model_normalized.obj").write_text(
        "mtllib color.mtl\nv 9 19 30\nv 11 19 30\nv 11 21 30\nv 9 21 30\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl color\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    (mesh / "color.mtl").write_text("newmtl color\nKa 1 1 1\nKd 1 1 1\nmap_Kd color.png\n")
    Image.fromarray(np.full((2, 2, 3), [200, 80, 40], dtype=np.uint8)).save(mesh / "color.png")
    prepare(Namespace(root=str(root), output=str(output), split=None, views=2, num_points=16,
                      image_size=16, distance=2.7, elevation=15., background=0.,
                      texture_atlas_size=8, device="cpu", shard=None,
                      seed=0, limit=None, overwrite=False))
    with np.load(output / "03001627/fixture.npz") as archive:
        assert np.linalg.norm(archive["points"], axis=-1).max() <= 1.00001
        np.testing.assert_allclose(archive["points"][:, 2], 0, atol=1e-6)
        np.testing.assert_allclose(archive["K"][:, :2, 2], 7.5)
        assert (archive["images"].max(axis=(1, 2, 3)) > 0).all()
    sample = MultiViewPointCloudDataset(output, 16, 2, 16)[0]
    assert sample["images"].shape == (2, 3, 16, 16)
    assert json.loads((output / "preparation.json").read_text())["failures"] == []


@pytest.mark.skipif(importlib.util.find_spec("pytorch3d") is None, reason="PyTorch3D unavailable")
def test_mesh_preparation_colors_each_material_and_rejects_uncolored_meshes(tmp_path):
    from PIL import Image
    from prepare_shapenet import prepare

    root = tmp_path / "meshes"
    quad = "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
    colored = root / "03001627" / "two_materials" / "models"
    colored.mkdir(parents=True)
    # One face uses a texture image and the other only a diffuse color. Loading
    # through TexturesUV painted both faces with the first (and only) image.
    (colored / "model_normalized.obj").write_text(
        "mtllib color.mtl\n" + quad + "usemtl image\nf 1/1 2/2 3/3\nusemtl flat\nf 1/1 3/3 4/4\n")
    (colored / "color.mtl").write_text("newmtl image\nKd 1 1 1\nmap_Kd red.png\nnewmtl flat\nKd 0 1 0\n")
    Image.fromarray(np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)).save(colored / "red.png")
    settings = dict(root=str(root), split=None, views=2, num_points=16, image_size=32, distance=2.7,
                    elevation=15., background=0., texture_atlas_size=8, device="cpu", seed=0, shard=None,
                    limit=None, overwrite=False)
    prepare(Namespace(output=str(tmp_path / "paired"), **settings))
    with np.load(tmp_path / "paired/03001627/two_materials.npz") as archive:
        pixels = archive["images"].reshape(-1, 3).astype(int)
    assert (pixels == [255, 0, 0]).all(-1).any() and (pixels == [0, 255, 0]).all(-1).any()
    uncolored = root / "03001627" / "no_materials" / "models"
    uncolored.mkdir(parents=True)
    (uncolored / "model_normalized.obj").write_text(quad + "f 1 2 3\nf 1 3 4\n")
    with pytest.raises(RuntimeError, match="1 meshes failed"):
        prepare(Namespace(output=str(tmp_path / "rejected"), **settings))
    failures = json.loads((tmp_path / "rejected/preparation.json").read_text())["failures"]
    assert len(failures) == 1 and "no materials" in failures[0]["error"]


@pytest.mark.skipif(importlib.util.find_spec("pytorch3d") is None, reason="PyTorch3D unavailable")
def test_mesh_preparation_uses_diffuse_color_for_broken_textures(tmp_path):
    from prepare_shapenet import prepare

    # ShapeNet materials sometimes name a folder or a corrupt image as texture.
    mesh = tmp_path / "meshes/03001627/broken/models"
    mesh.mkdir(parents=True)
    (mesh / "model_normalized.obj").write_text(
        "mtllib broken.mtl\nv -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl folder\nf 1/1 2/2 3/3\nf 3/3 2/2 1/1\nusemtl corrupt\nf 1/1 3/3 4/4\nf 4/4 3/3 1/1\n")
    (mesh / "broken.mtl").write_text("newmtl folder\nKd 1 0 0\nmap_Kd ../\n"
                                     "newmtl corrupt\nKd 0 0 1\nmap_Kd corrupt.png\n")
    (mesh / "corrupt.png").write_bytes(b"not an image")
    prepare(Namespace(root=str(tmp_path / "meshes"), output=str(tmp_path / "paired"), split=None, views=2,
                      num_points=16, image_size=32, distance=2.7, elevation=15., background=0.,
                      texture_atlas_size=8, device="cpu", seed=0, limit=None, shard=None, overwrite=False))
    with np.load(tmp_path / "paired/03001627/broken.npz") as archive:
        pixels = archive["images"].reshape(-1, 3).astype(int)
    assert (pixels == [255, 0, 0]).all(-1).any() and (pixels == [0, 0, 255]).all(-1).any()


@pytest.mark.skipif(importlib.util.find_spec("pytorch3d") is None, reason="PyTorch3D unavailable")
def test_mesh_preparation_shows_each_side_of_double_sided_faces(tmp_path):
    from prepare_shapenet import prepare, wrapped_material_atlas
    from pytorch3d.io.mtl_io import make_material_atlas

    image, uvs = torch.rand(8, 8, 3), torch.rand(5, 3, 2)
    torch.testing.assert_close(wrapped_material_atlas(image, uvs, 4), make_material_atlas(image, uvs, 4))
    # ShapeNet v2 stores both windings of a surface, often with different
    # materials. Without back-face culling the two copies z-fight.
    mesh = tmp_path / "meshes/03001627/two_sided/models"
    mesh.mkdir(parents=True)
    (mesh / "model_normalized.obj").write_text(
        "mtllib sides.mtl\nv -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
        "usemtl front\nf 1 2 3\nf 1 3 4\nusemtl back\nf 1 3 2\nf 1 4 3\n")
    (mesh / "sides.mtl").write_text("newmtl front\nKd 1 0 0\nnewmtl back\nKd 0 1 0\n")
    prepare(Namespace(root=str(tmp_path / "meshes"), output=str(tmp_path / "paired"), split=None, views=2,
                      num_points=16, image_size=32, distance=2.7, elevation=15., background=0.,
                      texture_atlas_size=8, device="cpu", seed=0, limit=None, shard=None, overwrite=False))
    with np.load(tmp_path / "paired/03001627/two_sided.npz") as archive:
        front, back = archive["images"].reshape(2, -1, 3).astype(int)
        normals, colors, visible = archive["normals"], archive["colors"], archive["visibility"]
        silhouettes = (archive["depths"] > 0) == (archive["images"].max(-1) > 0)
    red, green = np.array([255, 0, 0]), np.array([0, 255, 0])
    assert (front == red).all(-1).sum() > 50 and not (front == green).all(-1).any()
    assert (back == green).all(-1).sum() > 50 and not (back == red).all(-1).any()
    # Every point is seen from both sides; its normal and color describe one side.
    assert visible.shape == (2, 16) and visible.all() and silhouettes.all()
    np.testing.assert_allclose(np.abs(normals), [[0, 0, 1]] * 16, atol=1e-5)
    assert ((normals[:, 2] > 0) == (colors == red).all(-1)).all()
    assert ((colors == red).all(-1) | (colors == green).all(-1)).all()
