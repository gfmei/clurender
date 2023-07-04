#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/27/2023 1:44 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : o3d_utils.py
# @Software: PyCharm
import open3d as o3d
import torch
import numpy as np


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def make_point_cloud(pts, normals=None, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(pts))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(to_array(normals))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(to_array(color))
    return pcd