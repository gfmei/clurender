#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/22/2023 11:51 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : gen_data.py
# @Software: PyCharm
import os
# add path for demo utils functions
import sys
from pathlib import Path

import numpy as np
import pytorch3d
import torch
from pytorch3d.datasets import (
    ShapeNetCore,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)

SYNSET_DICT_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath('../'))
from lib.lib_utils import save_pcd


def sh_render(render, image_size, num_views=24, elevation_range=(-15, 15), azimuth_range=(0, 360),
              distance=1.0, device='cpu'):
    # Sample viewpoints
    elevations = torch.linspace(*elevation_range, num_views)
    azimuths = torch.linspace(*azimuth_range, num_views)
    elevations, azimuths = torch.meshgrid(elevations, azimuths)
    elevations = elevations.flatten()
    azimuths = azimuths.flatten()
    R, T = look_at_view_transform(distance, elevations, azimuths, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=image_size, cull_backfaces=True, )
    lights = PointLights(diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),),
                         location=torch.tensor([0.0, 5.0, -10.0], device=device)[None], device=device)
    images = render(
        model_ids=[model_id],
        device=device,
        cameras=cameras,
        raster_settings=raster_settings,
        lights=lights,
    )

    return images


if __name__ == '__main__':
    from PIL import Image

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    data_dir = '/home/gmei/Data/data/ShapeNetCore.v2'
    result_path = '/home/gmei/Data/data/ShapeNetCoreRender'
    shapenet_dataset = ShapeNetCore(data_dir, version=2, load_textures=True)
    raster_settings = RasterizationSettings(image_size=64, cull_backfaces=True, bin_size=0)
    lights = PointLights(diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),),
                         location=torch.tensor([0.0, 5.0, -10.0], device=device)[None], device=device)

    num_ids = range(0, shapenet_dataset.__len__())
    views = range(0, 360, 45)
    model_dir = "models/model_normalized.obj"
    for ids in num_ids:
        try:
            obj_dict = shapenet_dataset._get_item_ids(ids)
            class_id = obj_dict['synset_id']
            model_id = obj_dict['model_id']
            model_path = os.path.join(data_dir, class_id, model_id, model_dir)
            base_dir = os.path.join(result_path, class_id)
            if not os.path.isdir(base_dir):
                os.mkdir(base_dir)
            if not os.path.isdir(os.path.join(base_dir, model_id)):
                os.mkdir(os.path.join(base_dir, model_id))
            # Load the mesh object
            meshes = pytorch3d.io.load_objs_as_meshes([model_path])
            # Sample a point cloud
            points = pytorch3d.ops.sample_points_from_meshes(meshes, num_samples=4096).squeeze(0).cpu().numpy()
            point_cloud_path = os.path.join(base_dir, f"{model_id}/pcd.ply")
            save_pcd(points, point_cloud_path)
            for view in views:
                R, T = look_at_view_transform(1.0, 1.0, view)
                cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
                images_by_idxs = shapenet_dataset.render(
                    model_ids=[model_id],
                    device=device,
                    cameras=cameras,
                    raster_settings=raster_settings,
                    lights=lights,
                )
                image_path = os.path.join(base_dir, f"{model_id}/image_{view}.png")
                camera_path = os.path.join(base_dir, f"{model_id}/camera_{view}.pt")
                # save_obj(point_cloud_path, point_cloud)
                torch.save(cameras.state_dict(), camera_path)
                image_np = images_by_idxs.detach().cpu().numpy()[0]
                img = Image.fromarray(np.uint8(255 * image_np))  # no opencv required
                img.save(image_path)
        except Exception as e:
            print(e)
            pass
