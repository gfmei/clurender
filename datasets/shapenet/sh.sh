#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/20/2023 6:07 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : gen_train_data.py
# @Software: PyCharm
import os
import sys

import torch
from pytorch3d.structures import Pointclouds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))

from datasets.shapenet.render_shapenet import generate_tsfm, point_render, data_save
from datasets.shapenet.shapenet import load_shapenet_path, load_ply
from datasets.datautil import points_sampler, pc_normalize

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rot, trans = generate_tsfm(device)
    root = '/data/gmei/data'
    data_paths = load_shapenet_path(root)
    n_points = 4096
    for data_path in data_paths:
        print(data_path)
        points = load_ply(data_path)
        num = points.shape[0]
        if num > n_points:
            points = points_sampler(points, n_points)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        pcds = points
        points = torch.from_numpy(points).to(device)
        features = torch.ones_like(points, dtype=torch.float).to(device)
        points = Pointclouds(points=[points.to(device)], features=[features])
        # custom_color = (1.0, 0, 0)
        _, images, cameras = point_render(points, rot, trans, image_size=64, points_per_pixel=5, device=device)
        data_save(pcds, cameras, images, 's1', 's1', './')
        break