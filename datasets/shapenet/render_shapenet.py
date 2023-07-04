#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/20/2023 2:37 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : render_shapenet.py
# @Software: PyCharm
import os
from typing import Optional, List

import torch
from pytorch3d.datasets import BlenderCamera
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    HardPhongShader, RasterizationSettings,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from pytorch3d.structures import Meshes
from torch import nn
from torch.types import Device

from lib.lib_utils import save_pcd


def load_mesh(shapenet_path, class_id, model_id, device):
    model_path = os.path.join(shapenet_path, class_id, model_id, "models", "model_normalized.obj")
    verts, faces, _ = load_obj(model_path)
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    return mesh


def mesh_render(mesh, rot, trans, image_size=512, fov=45.0, blur_radius=0.0, near=0.01, far=100.0,
                faces_per_pixel=1, clip_barycentric_coords=True, num_points=10000):
    device = mesh.device
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=clip_barycentric_coords
    )
    cameras = PerspectiveCameras(
        R=rot.to(device),
        T=trans.to(device),
        device=device,
        fov=fov,
        near=near,
        far=far
    )
    # initialize points renderer
    points_renderer = PointsRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras
        )
    )
    # generate point cloud from mesh
    point_cloud = mesh.sample_points_poisson_disk(num_points)
    points = point_cloud.points_list()[0].to(device)
    # render images from point cloud
    image = points_renderer(points, point_cloud_normals=None)

    return image, point_cloud, cameras


def point_render(points, R, T, image_size=512, points_per_pixel=5, device='cpu'):
    # R, T = look_at_view_transform(dist=2.7, elev=0, azim=i * 15)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
    # Initialize the rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=1e-2,
        points_per_pixel=points_per_pixel,
    )
    # Initialize the renderer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(0, 0, 1))
    )
    # Render point cloud and save images, point cloud, and rendering parameters
    images = renderer(points)
    return points, images, cameras


def generate_tsfm(device):
    rot, transl = look_at_view_transform(2.732, 0, 0)

    return rot.to(device), transl.to(device)


def set_view(device):
    # Set up view transform
    cameras = nn.Parameter(torch.tensor([[0.0, 0.0, 2.732], [0.0, 0.0, 2.732]], device=device))
    R, T = look_at_view_transform(2.732, 0, 0)
    cameras[0, :3, :3] = R
    cameras[0, :3, 3] = T[0, 3, :3]
    return cameras


def data_save(point_cloud, cameras, image, class_id, model_id, result_path):
    from PIL import Image
    # save results
    image_path = os.path.join(result_path, f"{class_id}_{model_id}_image.jpeg")
    point_cloud_path = os.path.join(result_path, f"{class_id}_{model_id}_pcd.ply")
    camera_path = os.path.join(result_path, f"{class_id}_{model_id}_camera.pt")
    save_pcd(point_cloud, point_cloud_path)
    # save_obj(point_cloud_path, point_cloud)
    torch.save(cameras.state_dict(), camera_path)
    im = Image.fromarray(image.squeeze().cpu().numpy(), 'RGB')
    im.save(image_path)


class ShapeNetRD(ShapeNetBase):
    def __init__(self, root, n_points=1024, transform=False, fps=False):
        super().__init__()

    def render(
            self,
            model_ids: Optional[List[str]] = None,
            categories: Optional[List[str]] = None,
            sample_nums: Optional[List[int]] = None,
            idxs: Optional[List[int]] = None,
            view_idxs: Optional[List[int]] = None,
            shader_type=HardPhongShader,
            device: Device = "cpu",
            **kwargs,
    ) -> torch.Tensor:
        """
        Render models with BlenderCamera by default to achieve the same orientations as the
        R2N2 renderings. Also accepts other types of cameras and any of the args that the
        render function in the ShapeNetBase class accepts.
        Args:
            view_idxs: each model will be rendered with the orientation(s) of the specified
                views. Only render by view_idxs if no camera or args for BlenderCamera is
                supplied.
            Accepts any of the args of the render function in ShapeNetBase:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Shader to use for rendering. Examples include HardPhongShader
            (default), SoftPhongShader etc or any other type of valid Shader class.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports and any of the
                args that BlenderCamera supports.
        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        r = torch.cat([self[idxs[i], view_idxs]["R"] for i in range(len(idxs))])
        t = torch.cat([self[idxs[i], view_idxs]["T"] for i in range(len(idxs))])
        k = torch.cat([self[idxs[i], view_idxs]["K"] for i in range(len(idxs))])
        # Initialize default camera using R, T, K from kwargs or R, T, K of the specified views.
        blend_cameras = BlenderCamera(
            R=kwargs.get("R", r),
            T=kwargs.get("T", t),
            K=kwargs.get("K", k),
            device=device,
        )
        cameras = kwargs.get("cameras", blend_cameras).to(device)
        kwargs.pop("cameras", None)
        # pass down all the same inputs
        return super().render(
            idxs=idxs, shader_type=shader_type, device=device, cameras=cameras, **kwargs
        )


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
