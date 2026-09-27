"""Prepare calibrated paired archives from ShapeNetCore v2 textured meshes.

Each archive holds surface points with normals and colors, RGB views with
depth maps, the cameras, and which points each view sees. This provides a
reproducible PyTorch3D orbit-rendering alternative. It does not reproduce the
paper's DISN/Blender target images; existing calibrated renderings can instead
be packed directly into the documented NPZ format.
"""

import argparse
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from datasets.multiview import orbit_cameras, project_points


def wrapped_material_atlas(image, faces_verts_uvs, texture_size):
    """PyTorch3D's per-face texture sampling, wrapping UVs per sample.

    ShapeNet tiles textures with UVs outside [0, 1]. PyTorch3D wraps each
    vertex separately, so a face that spans a tile edge samples across the
    whole image. Interpolating first and then wrapping keeps it local. The
    atlas layout matches pytorch3d.io.mtl_io.make_material_atlas.
    """
    size = texture_size
    cells = torch.arange(size, device=faces_verts_uvs.device, dtype=faces_verts_uvs.dtype)
    y, x = torch.meshgrid(cells, cells, indexing="ij")
    grid = torch.stack((x, y), -1)
    below_diagonal = (grid.sum(-1) < size)[..., None]
    w01 = torch.where(below_diagonal, (grid + 1 / 3) / size, (size - 1 - grid + 2 / 3) / size)
    bary = torch.cat((w01, 1 - w01.sum(-1, keepdim=True)), -1)
    uv = (faces_verts_uvs[:, None, None] * bary[..., None]).sum(-2) % 1
    samples = F.grid_sample(image.permute(2, 0, 1)[None], (uv * 2 - 1).reshape(1, -1, size, 2),
                            mode="bilinear", align_corners=True)
    # Explicit channels: a material that no face uses has zero rows.
    return samples[0].permute(1, 2, 0).reshape(len(uv), size, size, image.shape[-1])


def texture_path_manager():
    """A PathManager that reports broken texture references as missing.

    Some ShapeNet materials name a folder (``map_Kd ../``) or a corrupt image.
    Reported as missing, PyTorch3D warns and uses the material's diffuse color.
    """
    from iopath.common.file_io import PathManager
    from PIL import Image

    class TexturePathManager(PathManager):
        def exists(self, path, **kwargs):
            if not super().exists(path, **kwargs) or os.path.isdir(path):
                return False
            if Path(path).suffix.lower() in (".obj", ".mtl"):
                return True
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, SyntaxError, ValueError):
                return False
            return True

    return TexturePathManager()


def with_back_faces(verts, faces, atlas):
    """Add a reversed copy of every face that lacks one.

    ShapeNet v2 stores each surface twice with opposite windings, often with a
    different material on each side, and the two copies z-fight unless back
    faces are culled. Completing single-sided faces lets the renderer cull
    back faces without opening holes. Swapping a face's first two vertices
    swaps its (w0, w1) barycentrics, so its atlas cell grid is transposed.
    """
    positions = torch.unique((verts * 1e5).round(), dim=0, return_inverse=True)[1]
    corners = positions[faces]

    def oriented(triangles):
        # One representation for each triangle and winding, in any rotation.
        rotations = torch.stack([triangles.roll(shift, dims=1) for shift in range(3)])
        first = rotations[..., 0].argmin(0)
        return rotations[first, torch.arange(len(triangles), device=triangles.device)]

    rows = torch.unique(torch.cat((oriented(corners), oriented(corners[:, [1, 0, 2]]))),
                        dim=0, return_inverse=True)[1]
    missing = ~torch.isin(rows[len(faces):], rows[:len(faces)])
    return (torch.cat((faces, faces[missing][:, [1, 0, 2]])),
            torch.cat((atlas, atlas[missing].transpose(1, 2))))


def point_attributes(points, normals, texture_colors, images, depths, transforms, intrinsics):
    """Per-view visibility, camera-facing normals and rendered colors of points.

    A point is visible in a view when the rendered depth at its pixel matches
    its own depth to within two pixel widths. ShapeNet surfaces are often
    double-sided with a different material per side, so each normal is turned
    toward the view that sees its point most frontally, and the color is the
    median of the pixels showing that side. Points that no view sees keep their
    sampled texture color, and their normals face away from the center.
    """
    views, height, width = depths.shape
    pixels, depth = project_points(points, transforms, intrinsics)
    column, row = pixels.round().long().unbind(-1)
    inside = (depth > 0) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    column, row = column.clamp(0, width - 1), row.clamp(0, height - 1)
    view = torch.arange(views, device=points.device)[:, None]
    rendered = depths[view, row, column]
    visible = inside & (rendered > 0) & ((rendered - depth).abs() <= 2 * depth / intrinsics[:, None, 0, 0])
    centers = -(transforms[:, :3, :3].transpose(1, 2) @ transforms[:, :3, 3:])[..., 0]
    facing = (F.normalize(centers[:, None] - points, dim=-1) * normals).sum(-1)
    best = torch.where(visible, facing.abs(), -1.).argmax(0, keepdim=True)
    sign = torch.where(visible.any(0), facing.gather(0, best)[0], (normals * points).sum(-1)).sign()
    sign = torch.where(sign == 0, 1., sign)
    same_side = visible & (facing * sign > 0)
    colors = torch.where(same_side[..., None], images[view, row, column].float(), float("nan"))
    colors = torch.where(same_side.any(0)[:, None], colors.nanmedian(0).values, texture_colors * 255)
    return normals * sign[:, None], colors.round().clamp(0, 255).to(torch.uint8), visible


def prepare(args):
    from pytorch3d.io import load_obj, mtl_io
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.renderer import (AmbientLights, BlendParams, HardPhongShader, MeshRasterizer,
                                   MeshRenderer, RasterizationSettings, TexturesAtlas)
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    if min(args.views, args.num_points, args.image_size, args.texture_atlas_size) < 1 or args.distance <= 1:
        raise ValueError("Counts must be positive and camera distance must exceed the unit object radius")
    if args.shard and not 0 <= args.shard[0] < args.shard[1]:
        raise ValueError("--shard INDEX COUNT requires 0 <= INDEX < COUNT")
    root, output = Path(args.root), Path(args.output)
    if args.split:
        # One ShapeNet category/model identifier per line, for a chosen train split.
        entries = Path(args.split).read_text().splitlines()
        paths = [root / entry.strip() / "models/model_normalized.obj" for entry in entries
                 if entry.strip() and not entry.lstrip().startswith("#")]
    else:
        paths = sorted(root.glob("*/*/models/model_normalized.obj"))
    if not paths:
        raise ValueError(f"No ShapeNetCore v2 meshes found in {root}")
    if args.limit:
        paths = paths[:args.limit]
    device = torch.device(args.device)
    transforms, intrinsics = orbit_cameras(args.views, args.image_size, args.distance, args.elevation)
    torch_transforms = torch.from_numpy(transforms).to(device)
    archive_K = torch.from_numpy(intrinsics).to(device)
    torch_K = archive_K.clone()
    # PyTorch3D screen coordinates use half-integer pixel centers. Archives
    # use integer pixel centers, so shift principal points at this boundary.
    torch_K[:, :2, 2] += 0.5
    cameras = cameras_from_opencv_projection(
        R=torch_transforms[:, :3, :3], tvec=torch_transforms[:, :3, 3],
        camera_matrix=torch_K,
        image_size=torch.full((args.views, 2), args.image_size, device=device))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(),  # Settings depend on each mesh; see below.
        shader=HardPhongShader(device=device, lights=AmbientLights(device=device),
                               blend_params=BlendParams(background_color=(args.background,) * 3)))
    output.mkdir(parents=True, exist_ok=True)
    path_manager = texture_path_manager()
    failures, written = [], 0
    with torch.no_grad():
        for index, path in enumerate(paths):
            if args.shard and index % args.shard[1] != args.shard[0]:
                continue
            identifier = path.parent.parent.relative_to(root)
            destination = output / identifier.with_suffix(".npz")
            if destination.exists() and not args.overwrite:
                continue
            # Stable per-object sampling independent of skipped existing files.
            torch.manual_seed(args.seed + index)
            try:
                # A per-face atlas colors each face from its own material: a texture
                # image or a plain diffuse (Kd) color. PyTorch3D's UV path applies
                # only the first texture image to every face of the mesh.
                with mock.patch.object(mtl_io, "make_material_atlas", wrapped_material_atlas):
                    verts, faces, aux = load_obj(str(path), device=device, load_textures=True,
                                                 create_texture_atlas=True, texture_wrap=None,
                                                 texture_atlas_size=args.texture_atlas_size,
                                                 path_manager=path_manager)
                if not aux.material_colors and not aux.texture_images:
                    raise ValueError("Mesh has no materials; photometric supervision would be missing")
                face_indices, atlas = with_back_faces(verts, faces.verts_idx, aux.texture_atlas.to(device))
                mesh = Meshes(verts=[verts], faces=[face_indices], textures=TexturesAtlas(atlas=[atlas]))
                vertices = mesh.verts_packed()
                center = (vertices.max(0).values + vertices.min(0).values) / 2
                mesh.offset_verts_(-center.expand_as(vertices))
                scale = mesh.verts_packed().norm(dim=-1).max().clamp_min(1e-8)
                # PyTorch3D accepts a scalar Python float or one scale per
                # mesh. A zero-dimensional tensor cannot be indexed by mesh.
                mesh.scale_verts_(scale.reciprocal().reshape(1))
                points, normals, texture_colors = (value[0] for value in sample_points_from_meshes(
                    mesh, args.num_points, return_normals=True, return_textures=True))
                # PyTorch3D's default per-bin face limit drops faces of detailed
                # meshes on the GPU; allowing every face keeps renders complete.
                raster_settings = RasterizationSettings(
                    image_size=args.image_size, blur_radius=0, faces_per_pixel=1, cull_backfaces=True,
                    max_faces_per_bin=len(face_indices))
                images, depths = [], []
                for view in range(args.views):
                    fragments = renderer.rasterizer(mesh, cameras=cameras[view], raster_settings=raster_settings)
                    rgba = renderer.shader(fragments, mesh, cameras=cameras[view])[0]
                    images.append((rgba[..., :3].clamp(0, 1) * 255).round().to(torch.uint8))
                    depths.append(fragments.zbuf[0, ..., 0].clamp_min(0))  # Background is 0.
                images, depths = torch.stack(images), torch.stack(depths)
                normals, colors, visibility = point_attributes(
                    points, normals, texture_colors, images, depths, torch_transforms, archive_K)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream, points=points.cpu().numpy(), normals=normals.cpu().numpy(),
                        colors=colors.cpu().numpy(), visibility=visibility.cpu().numpy(),
                        images=images.cpu().numpy(), depths=depths.cpu().numpy().astype(np.float16),
                        tsfms=transforms, K=intrinsics)
                os.replace(temporary, destination)
                written += 1
                print(f"[{index + 1}/{len(paths)}] {destination}", flush=True)
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append({"mesh": str(path), "error": str(exc)})
                print(f"FAILED {path}: {exc}", flush=True)
    report = output / ("preparation-{}-of-{}.json".format(*args.shard) if args.shard else "preparation.json")
    report.write_text(json.dumps({"args": vars(args), "written": written, "failures": failures}, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} meshes failed; see {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", help="Text file of category/model identifiers")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--distance", type=float, default=2.7)
    parser.add_argument("--elevation", type=float, default=15)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--texture-atlas-size", type=int, default=8,
                        help="Per-face texture samples along each side")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", type=int, nargs=2, metavar=("INDEX", "COUNT"),
                        help="Process only meshes whose position in the list is INDEX modulo COUNT")
    parser.add_argument("--overwrite", action="store_true")
    prepare(parser.parse_args())
