"""Regression cases found while reviewing the pretraining implementation."""

import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from main_pretrain import (parser, reduce_parallel_losses, resolve_args, train,
                           validate_parallel_batch)
from models.clurender import CluRender, balanced_assign
from models.common import farthest_point_sample, index_points, knn, square_distance
from models.dgcnn import DGCNN
from models.losses import BalancedClustering, round_transport_plan
from models.pointnet import PointNet, STN3d, STNkd
from models.renderer import PointRenderer, RenderConfig
from datasets.multiview import orbit_cameras, write_smoke_dataset


@pytest.fixture(autouse=True)
def setup_torch():
    torch.set_num_threads(2)
    torch.manual_seed(6)


def test_fps_uses_only_selected_points_and_never_repeats():
    points = torch.zeros(1, 5, 3)
    points[0, :, 0] = torch.arange(5)
    indices = farthest_point_sample(points, 5, is_center=True)
    # After the two endpoints, the middle is farthest from the selected set.
    assert indices[0, :3].tolist() == [0, 4, 2]
    assert indices.unique().numel() == 5
    indices = farthest_point_sample(torch.zeros(2, 5, 3), 5, is_center=True)
    assert all(row.unique().numel() == 5 for row in indices)
    with pytest.raises(ValueError, match="npoint"):
        farthest_point_sample(points, 6)


def test_geometric_distances_and_knn_survive_large_world_translation():
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]]])
    translated = (points + 1e5).requires_grad_()
    distances = square_distance(translated, translated)
    expected = torch.tensor([[[0., 1., 9.], [1., 0., 4.], [9., 4., 0.]]])
    torch.testing.assert_close(distances, expected)
    assert knn(translated.transpose(1, 2), 2)[0, 0].tolist() == [0, 1]
    distances.sum().backward()
    torch.testing.assert_close(translated.grad[..., 0], torch.tensor([[-16., -4., 20.]]))


@pytest.mark.parametrize("scale", [1., 10., 100.])
def test_default_iteration_budget_produces_feasible_clustering_labels(scale):
    net = BalancedClustering(32, 16)
    features = torch.randn(2, 32, 128, requires_grad=True)
    loss, details = net(features, scale * torch.randn(2, 3, 128), return_details=True)
    labels = details["labels"]
    assert labels.min() >= 0 and labels.max() <= 1 + 1e-5
    torch.testing.assert_close(labels.sum(-1), torch.ones(2, 128), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(labels.sum(1), torch.full((2, 16), 8.), atol=2e-5, rtol=2e-6)
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.norm() > 0


def test_transport_rounding_handles_zero_rows_and_already_balanced_plan():
    p = torch.tensor([[.25, .75]], dtype=torch.float64)
    q = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    for plan in (torch.zeros(1, 2, 3, dtype=p.dtype), p.unsqueeze(-1) * q.unsqueeze(-2),
                 torch.tensor([[[0., 0., 0.], [1., 2., 3.]]], dtype=p.dtype)):
        rounded = round_transport_plan(plan, p, q)
        torch.testing.assert_close(rounded.sum(-1), p)
        torch.testing.assert_close(rounded.sum(-2), q)
        assert torch.isfinite(rounded).all() and rounded.min() >= 0


def test_raw_assignments_remain_available_for_comparison():
    raw = BalancedClustering(32, 16, round_assignments=False)
    _, details = raw(torch.randn(2, 32, 128), 10 * torch.randn(2, 3, 128), return_details=True)
    assert details["unrounded_marginal_error"] > .1
    assert (details["labels"].sum(-1) - 1).abs().max() > .1


def test_pointnet_input_transform_trains_without_feature_transform():
    model = PointNet(dims=32, feature_transform=False).eval()
    model(torch.randn(2, 3, 8))[0].square().mean().backward()
    assert model.stn.fc3.weight.grad is not None
    assert model.stn.fc3.weight.grad.norm() > 0


def test_pointnet_concatenated_features_support_nondefault_dimension():
    model = PointNet(dims=32, feat_type="concat").eval()
    features, point_features, transform = model(torch.randn(2, 3, 8))
    assert features.shape == (2, 96, 8) and point_features.shape == (2, 32, 8)
    assert transform is None


@pytest.mark.parametrize("network,channels", [(STN3d(3), 3), (STNkd(8), 8)])
def test_pointnet_transforms_preserve_dtype(network, channels):
    result = network.double().eval()(torch.randn(2, channels, 8, dtype=torch.float64))
    assert result.dtype == torch.float64 and torch.isfinite(result).all()


def test_parallel_reduction_weights_losses_and_gradients_by_samples():
    values = torch.tensor([2., 10.], requires_grad=True)
    reduced = reduce_parallel_losses({"loss": values, "batch_size": torch.tensor([3., 1.]),
                                       "assignment_error": torch.tensor([.5, 2.])})
    torch.testing.assert_close(reduced["loss"], torch.tensor(4.))
    assert reduced["assignment_error"] == 2
    reduced["loss"].backward()
    torch.testing.assert_close(values.grad, torch.tensor([.75, .25]))


def test_parallel_pointnet_rejects_actual_singleton_tail():
    with pytest.raises(ValueError, match="every GPU shard"):
        validate_parallel_batch(10, 4, "pointnet")
    validate_parallel_batch(6, 4, "pointnet")  # 2,2,2: PyTorch produces only 3 chunks.
    validate_parallel_batch(10, 4, "dgcnn")


def test_resume_preserves_execution_settings_unless_explicitly_overridden(tmp_path):
    original = resolve_args(parser().parse_args([
        "--root", str(tmp_path), "--device", "cuda:1", "--workers", "7", "--threads", "3",
        "--log-every", "2", "--save-every", "4", "--no-checkpoint-rendering",
    ]))
    checkpoint = {"args": vars(original)}
    resumed = resolve_args(parser().parse_args(["--resume", "saved.pth"]), checkpoint)
    for name in ("device", "workers", "threads", "log_every", "save_every", "checkpoint_rendering"):
        assert getattr(resumed, name) == getattr(original, name)
    explicit = resolve_args(parser().parse_args([
        "--resume", "saved.pth", "--device", "cpu", "--workers", "0", "--checkpoint-rendering",
    ]), checkpoint)
    assert explicit.device == "cpu" and explicit.workers == 0 and explicit.checkpoint_rendering
    legacy = copy.deepcopy(checkpoint)
    del legacy["args"]["round_assignments"]
    assert not resolve_args(parser().parse_args(["--resume", "saved.pth"]), legacy).round_assignments


def test_completed_resume_returns_real_checkpoint_without_creating_output(tmp_path):
    checkpoint_path = tmp_path / "done.pth"
    args = resolve_args(parser().parse_args(["--smoke", "--output", str(tmp_path / "old"), "--epochs", "2"]))
    torch.save({"args": vars(args), "epoch": 2}, checkpoint_path)
    requested_output = tmp_path / "new"
    result = train(parser().parse_args(["--resume", str(checkpoint_path), "--output", str(requested_output)]))
    assert result == checkpoint_path and result.is_file()
    assert not requested_output.exists()


def test_smoke_does_not_create_data_under_an_explicit_missing_root(tmp_path):
    missing = tmp_path / "misspelled_data_path"
    with pytest.raises(FileNotFoundError):
        train(parser().parse_args(["--smoke", "--root", str(missing), "--output", str(tmp_path / "run")]))
    assert not missing.exists()


@pytest.mark.parametrize("flag,value", [("--lr", "nan"), ("--epsilon", "inf"),
                                         ("--orthogonal-weight", "-1"), ("--color-dims", "0"),
                                         ("--sinkhorn-tolerance", "nan"), ("--render-weight", "-1")])
def test_invalid_settings_fail_early(tmp_path, flag, value):
    values = [value] * 6 if flag == "--color-dims" else [value]
    with pytest.raises(ValueError):
        resolve_args(parser().parse_args(["--root", str(tmp_path), flag, *values]))


def test_renderer_optional_diagnostics_and_occupancy_agree():
    renderer = PointRenderer(RenderConfig(render_size=1, points_per_pixel=8, backend="torch"))
    points, colors = torch.tensor([[[0., 0., 1.]]]), torch.ones(1, 1, 3)
    complete, compact = renderer(points, colors), renderer(points, colors, return_raster=False)
    assert "raster_output" not in compact
    assert complete["raster_output"]["idx"].shape == (1, 8, 1, 1)
    for key in compact:
        torch.testing.assert_close(compact[key], complete[key])
    assert compact["valid_pts"].item() == 1 / 8


def test_fractional_blending_exponent_has_finite_zero_weight_gradients():
    renderer = PointRenderer(RenderConfig(render_size=3, weight_calculation="linear",
                                          eta=.5, backend="torch", radius=.1))
    points = torch.tensor([[[0., 0., 1.]]], requires_grad=True)
    image = renderer(points, torch.ones(1, 1, 3))["feats"]
    image.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_checkpointed_rendering_preserves_loss_gradients_and_saves_memory():
    standard = CluRender(DGCNN(32, 4), 32, 4, RenderConfig(render_size=16, backend="torch"),
                         color_dims=(32, 24, 16, 16, 12, 8))
    recomputed = copy.deepcopy(standard)
    recomputed.checkpoint_rendering = True
    transforms, K = orbit_cameras(3, 16)
    args = (torch.rand(2, 3, 32) - .5, torch.rand(2, 3, 3, 16, 16),
            torch.from_numpy(transforms).unsqueeze(0).repeat(2, 1, 1, 1),
            torch.from_numpy(K).unsqueeze(0).repeat(2, 1, 1, 1))

    def forward_and_backward(model):
        saved_bytes = 0

        def pack(tensor):
            nonlocal saved_bytes
            saved_bytes += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = model(*args)
        loss.backward()
        return loss, saved_bytes

    loss_a, memory_a = forward_and_backward(standard)
    loss_b, memory_b = forward_and_backward(recomputed)
    torch.testing.assert_close(loss_a, loss_b)
    for a, b in zip(standard.parameters(), recomputed.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    assert memory_b < memory_a


def test_default_sinkhorn_budget_converges_once_clusters_separate():
    # On a trained checkpoint, the former fixed 20 iterations left most labels unconverged.
    defaults = parser().parse_args([])
    points = torch.rand(1, 1024, 3) * torch.tensor([1., 1.6, .8]) - torch.tensor([.5, .8, .4])
    points = points / points.norm(dim=-1).max()
    centers = index_points(points, farthest_point_sample(points, defaults.num_clusters, is_center=True))
    logits = 20 * F.one_hot(square_distance(points, centers).argmin(-1), defaults.num_clusters).float()
    cluster = BalancedClustering(defaults.num_clusters, defaults.num_clusters, defaults.epsilon,
                                 defaults.sinkhorn_iterations, tolerance=defaults.sinkhorn_tolerance)
    cluster.head = torch.nn.Identity()  # Voronoi scores give separated, unbalanced prototypes.
    _, details = cluster(logits.transpose(1, 2), points.transpose(1, 2), return_details=True)
    # Small slack: the stopping check and this report round differently.
    assert details["unrounded_marginal_error"] <= defaults.sinkhorn_tolerance + 1e-4


def test_clustering_objective_ignores_world_translation():
    cluster = BalancedClustering(16, 4)
    features, points = torch.randn(2, 16, 64), torch.rand(2, 3, 64) - .5
    loss, details = cluster(features, points, return_details=True)
    moved, moved_details = cluster(features, points + torch.tensor([3., -2., 5.]).view(1, 3, 1),
                                   return_details=True)
    torch.testing.assert_close(moved_details["orthogonal"], details["orthogonal"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(moved, loss, atol=1e-4, rtol=1e-4)


def test_loss_weights_scale_their_terms_and_zero_removes_the_branch():
    transforms, K = orbit_cameras(2, 8)
    inputs = (torch.rand(2, 3, 32) - .5, torch.rand(2, 2, 3, 8, 8),
              torch.from_numpy(transforms).unsqueeze(0).repeat(2, 1, 1, 1),
              torch.from_numpy(K).unsqueeze(0).repeat(2, 1, 1, 1))
    for render_weight, cluster_weight in ((.25, 1.), (0., 1.), (1., .5), (1., 0.)):
        model = CluRender(DGCNN(32, 4), 32, 4, RenderConfig(render_size=8, backend="torch"),
                          color_dims=(32, 24, 16, 16, 12, 8), render_weight=render_weight,
                          cluster_weight=cluster_weight)
        assert (model.color is None) == (render_weight == 0) and (model.cluster is None) == (cluster_weight == 0)
        losses = model(*inputs, return_details=True)
        torch.testing.assert_close(losses["loss"], cluster_weight * losses["clustering"]
                                   + render_weight * losses["rendering"] + losses["transform"])
        if render_weight == 0:
            assert losses["rendering"] == 0 and losses["coverage"] == 0
        if cluster_weight == 0:
            assert losses["clustering"] == 0 and losses["assignment_error"] == 0
        losses["loss"].backward()
        # DistributedDataParallel fails on parameters that never receive a gradient.
        assert all(p.grad is not None for p in model.parameters())
    for weights in ({"render_weight": -1.}, {"cluster_weight": float("nan")},
                    {"render_weight": 0., "cluster_weight": 0.}):
        with pytest.raises(ValueError, match="weight"):
            CluRender(DGCNN(32, 4), 32, 4, RenderConfig(render_size=8, backend="torch"), **weights)
    with pytest.raises(ValueError, match="loss-weight"):
        resolve_args(parser().parse_args(["--smoke", "--render-weight", "0", "--cluster-weight", "0"]))


def test_legacy_ot_targets_give_each_point_unit_mass():
    labels = balanced_assign(torch.randn(2, 3, 50), torch.randn(2, 3, 5), dst="eu")
    assert labels.shape == (2, 5, 50) and labels.min() >= 0
    torch.testing.assert_close(labels.sum(1), torch.ones(2, 50))
    torch.testing.assert_close(labels.sum(2), torch.full((2, 5), 10.))


def test_resume_keeps_legacy_sinkhorn_budget_and_accepts_step_cap(tmp_path):
    original = resolve_args(parser().parse_args(["--root", str(tmp_path), "--steps-per-epoch", "5"]))
    checkpoint = {"args": vars(original)}
    resumed = resolve_args(parser().parse_args(["--resume", "saved.pth", "--steps-per-epoch", "2"]), checkpoint)
    assert resumed.steps_per_epoch == 2 and resumed.sinkhorn_tolerance == original.sinkhorn_tolerance
    legacy = copy.deepcopy(checkpoint)
    del legacy["args"]["sinkhorn_tolerance"], legacy["args"]["render_weight"], legacy["args"]["cluster_weight"]
    resumed = resolve_args(parser().parse_args(["--resume", "saved.pth"]), legacy)
    assert resumed.sinkhorn_tolerance == 0 and resumed.render_weight == 1 and resumed.cluster_weight == 1


def test_auto_device_is_saved_unresolved_so_resume_can_change_nodes(tmp_path):
    write_smoke_dataset(tmp_path / "data")
    checkpoint_path = train(parser().parse_args([
        "--root", str(tmp_path / "data"), "--output", str(tmp_path / "run"), "--device", "auto",
        "--renderer", "torch", "--emb-dims", "32", "--k", "4", "--num-points", "32", "--num-views", "2",
        "--num-clusters", "4", "--image-size", "16", "--color-dims", "32", "24", "16", "16", "12", "8",
        "--batch-size", "2", "--points-per-pixel", "8", "--workers", "0", "--epochs", "1"]))
    assert torch.load(checkpoint_path, weights_only=True)["args"]["device"] == "auto"


def test_texture_atlas_wraps_each_sample_not_each_vertex():
    from prepare_shapenet import wrapped_material_atlas

    # Thirds of the texture are red, green and blue. A face spanning u = 0.8..1.2
    # covers blue then red; wrapping its vertices instead would span 0..0.8.
    image = torch.zeros(4, 30, 3)
    image[:, :10, 0], image[:, 10:20, 1], image[:, 20:, 2] = 1, 1, 1
    uvs = torch.tensor([[[.8, .5], [1.2, .5], [1., .5]]])
    atlas = wrapped_material_atlas(image, uvs, 8)
    assert atlas.shape == (1, 8, 8, 3)
    assert atlas[..., 1].max() < 1e-6 and atlas[..., 0].max() > .5 and atlas[..., 2].max() > .5
    # ShapeNet meshes can define a textured material that no face uses.
    assert wrapped_material_atlas(image, torch.zeros(0, 3, 2), 8).shape == (0, 8, 8, 3)


def test_single_sided_faces_get_reversed_twins_with_transposed_atlas():
    from prepare_shapenet import with_back_faces

    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.], [1., 1., 1.]])
    # Face 1 reverses face 0 through a rotation and a duplicated vertex; face 2 is single-sided.
    faces = torch.tensor([[0, 1, 2], [2, 1, 3], [0, 1, 4]])
    atlas = torch.rand(3, 4, 4, 3)
    completed, completed_atlas = with_back_faces(verts, faces, atlas)
    assert completed.tolist() == faces.tolist() + [[1, 0, 4]]
    torch.testing.assert_close(completed_atlas[3], atlas[2].transpose(0, 1))
    again, _ = with_back_faces(verts, completed, completed_atlas)
    assert len(again) == len(completed)


def test_point_attributes_use_visibility_to_orient_normals_and_pick_colors():
    from datasets.multiview import project_points
    from prepare_shapenet import point_attributes

    transforms, K = (torch.from_numpy(array) for array in orbit_cameras(1, 15))
    center = -(transforms[0, :3, :3].T @ transforms[0, :3, 3])
    forward = -center / center.norm()
    # Two points on the ray through the principal point; the far one is hidden.
    points = torch.stack((torch.zeros(3), center + 3.5 * forward))
    pixels, depth = project_points(points, transforms, K)
    torch.testing.assert_close(pixels[0], K[0, :2, 2].expand(2, 2))
    depths = torch.zeros(1, 15, 15)
    depths[0, 7, 7] = depth[0, 0]
    images = torch.zeros(1, 15, 15, 3, dtype=torch.uint8)
    images[0, 7, 7] = torch.tensor([10, 20, 30])
    normals = forward.expand(2, 3)  # Both face away from the camera.
    texture = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    oriented, colors, visible = point_attributes(points, normals, texture, images, depths, transforms, K)
    assert visible.tolist() == [[True, False]]
    torch.testing.assert_close(oriented, torch.stack((-forward, forward)))
    assert colors.tolist() == [[10, 20, 30], [0, 255, 0]]


def test_sliced_loss_keeps_only_a_sample_sized_buffer_and_matches_autograd():
    from models.losses import SlicedImageLoss, _SlicedSquaredW2

    x, y = torch.rand(2, 300, 5, dtype=torch.float64, requires_grad=True), torch.rand(2, 300, 5, dtype=torch.float64)
    directions = torch.nn.functional.normalize(torch.randn(40, 5, dtype=torch.float64), dim=-1)
    reference = ((x @ directions.T).sort(1).values - (y @ directions.T).sort(1).values).square().mean(
        dim=(0, 1)).sum() / len(directions)
    expected, = torch.autograd.grad(reference, x)
    loss = _SlicedSquaredW2.apply(x, y, directions, 16)
    actual, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(actual, expected)
    # Half-precision sort keys: the same loss and nearly the same gradient.
    approximate = _SlicedSquaredW2.apply(x, y, directions, 16, torch.float16)
    gradient, = torch.autograd.grad(approximate, x)
    torch.testing.assert_close(approximate, reference, rtol=1e-3, atol=0)
    assert torch.nn.functional.cosine_similarity(gradient.flatten(), expected.flatten(), 0) > .99

    saved = []
    rendered, target = torch.rand(2, 3, 32, 32, requires_grad=True), torch.rand(2, 3, 32, 32)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel() * t.element_size()) or t,
                                                  lambda t: t):
        SlicedImageLoss(64, chunk_size=16)(rendered, target).backward()
    # One (B, H*W, 5) float buffer; storing sort indices would take 2 x 1024 x 64 x 8 bytes.
    assert sum(saved) <= 2 * 32 * 32 * 5 * 4 and rendered.grad.abs().sum() > 0


def test_distributed_training_writes_one_checkpoint_and_log(tmp_path):
    import torch.multiprocessing as mp

    import socket

    args = parser().parse_args(["--smoke", "--output", str(tmp_path / "run"), "--root", str(tmp_path / "data"),
                                "--epochs", "1"])
    write_smoke_dataset(tmp_path / "data")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_distributed_worker, args=(2, port, args), nprocs=2, join=True)
    state = torch.load(tmp_path / "run/last.pth", weights_only=True)
    # Smoke settings: 4 samples and a global batch of 2, so one sample per process and step.
    assert state["epoch"] == 1 and state["global_step"] == 2
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 1


def _distributed_worker(rank, world, port, args):
    import os

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    train(args)


def test_early_stopping_counts_checks_without_sufficient_improvement():
    from main_pretrain import early_stopping

    state = {"best": float("-inf"), "stale": 0}
    assert early_stopping(state, .50, .01, 2) == (True, False)
    assert early_stopping(state, .505, .01, 2) == (False, False)  # Below min-delta.
    assert early_stopping(state, .52, .01, 2) == (True, False) and state == {"best": .52, "stale": 0}
    assert early_stopping(state, .40, .01, 2) == (False, False)
    assert early_stopping(state, .41, .01, 2) == (False, True)
    assert early_stopping({"best": .5, "stale": 9}, .1, .01, None) == (False, False)


def test_svm_monitor_stops_pretraining_early_and_resume_respects_it(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(0)
    labels = np.arange(30) % 3
    clouds = rng.normal(size=(30, 64, 3)).astype(np.float32)
    clouds[..., 0] *= 1 + 2 * labels[:, None]
    with h5py.File(tmp_path / "ply_data_train0.h5", "w") as archive:
        archive["data"], archive["label"] = clouds, labels[:, None].astype(np.uint8)
    # min-delta 1 makes every check after the first one count as stale.
    common = ["--smoke", "--output", str(tmp_path / "run"), "--monitor-svm", str(tmp_path),
              "--monitor-every", "1", "--patience", "1", "--min-delta", "0.999"]
    train(parser().parse_args(common + ["--epochs", "4"]))
    metrics = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in metrics] == [1, 2] and all("svm_val_accuracy" in row for row in metrics)
    assert (tmp_path / "run/best_svm.pth").is_file() and (tmp_path / "run/backbone_best_svm.pth").is_file()
    train(parser().parse_args(common + ["--epochs", "4", "--resume", str(tmp_path / "run/last.pth")]))
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_sinkhorn_cuda_graph_matches_eager_iterations():
    from models.common import sinkhorn

    cost = torch.rand(4, 256, 16, device="cuda", dtype=torch.float64)
    p = torch.full((4, 256), 1 / 256, device="cuda", dtype=torch.float64)
    q = torch.full((4, 16), 1 / 16, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        for anneal in (False, True):
            eager, _ = sinkhorn(cost, p, q, 1e-2, 1e-6, 300, check_every=10, anneal=anneal, use_graph=False)
            graphed, _ = sinkhorn(cost, p, q, 1e-2, 1e-6, 300, check_every=10, anneal=anneal)
            torch.testing.assert_close(graphed, eager)
