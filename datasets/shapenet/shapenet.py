#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/19/2023 8:44 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : shapenet.py
# @Software: PyCharm
import glob
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from torchvision.transforms import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from datasets.datautil import points_sampler, pc_normalize
import datasets.datautil as common

trans = transforms.Compose(
    [
        common.PointcloudSphereCrop(0.85),
        common.PointcloudToTensor(),
        common.PointcloudRandomInputDropout(p=1),
        common.PointcloudScale(lo=0.5, hi=2, p=1),
        common.PointcloudRotate(),
        common.PointcloudTranslate(0.5, p=1),
        common.PointcloudJitter(p=1),
        # common.PointcloudNormalize(True)
    ])


def load_ply(file_name: str, with_faces: bool = False, with_color: bool = False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def load_shapenet_path(data_dir):
    all_filepath = []
    for cls in glob.glob(os.path.join(data_dir, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath


class ShapeNet(Dataset):
    def __init__(self, root, n_points=1024, transform=False, fps=False):
        self.data = load_shapenet_path(root)
        self.n_points = n_points
        self.fps = fps
        self.transform = transform

    def __getitem__(self, item):
        pcd_path = self.data[item]
        points = load_ply(pcd_path)
        if self.transform:
            points = trans(points).numpy()
        num = points.shape[0]
        if num > self.n_points:
            points = points_sampler(points, self.n_points)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        return points

    def __len__(self):
        return len(self.data)


# Define dataset
class ShapeNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.pointcloud_paths = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".pt"):
                    self.pointcloud_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.pointcloud_paths)

    def __getitem__(self, idx):
        pointcloud_path = self.pointcloud_paths[idx]
        pointcloud = torch.load(pointcloud_path)
        pointclouds = Pointclouds(points=[pointcloud])
        rendering_dir = os.path.join(renderings_root, os.path.relpath(os.path.dirname(pointcloud_path), shapenet_root),
                                     os.path.splitext(os.path.basename(pointcloud_path))[0])
        images = []
        for i in range(24):
            image_path = os.path.join(rendering_dir, f"view{i:02d}_image.png")
            images.append(ToTensor()(Image.open(image_path)))
        images = torch.stack(images, dim=0)
        cameras = []
        for i in range(24):
            camera_path = os.path.join(rendering_dir, f"view{i:02d}_cameras.pt")
            cameras.append(torch.load(camera_path))
        return pointclouds, images, cameras


if __name__ == '__main__':
    root = '/data/gmei/data'
    dataset = ShapeNet(root, 1024, transform=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(train_loader):
        print(data.shape)
        break
