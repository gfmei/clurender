#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/19/2023 6:05 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : attnutils.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

from models.common import get_graph_feature


class PositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_dis = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_ang1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv_ang2 = nn.Sequential(
            nn.Conv1d(64, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        l_dim = dim // 2 + dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(l_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, points, k=5):
        """
        :param k: The number of neighbors
        :param points: [B, dim, N]
        :return:
        """
        centroid = torch.mean(points, dim=-1, keepdim=True)  # [B, dim, 1]
        p2gc = points - centroid  # [B, dim, num]
        g_dis = torch.square(p2gc).sum(dim=1, keepdim=True)
        dis_feature = self.conv_dis(g_dis)
        p2lc = get_graph_feature(points, k)[:, :3, :, :]  # [B, dim, num, k]
        p2gc_n = F.normalize(p2gc, dim=1)
        p2lc_n = F.normalize(p2lc, dim=1)
        alpha = torch.einsum('bdnk,bdn->bnk', p2lc_n, p2gc_n).unsqueeze(1)
        ang_feature = self.conv_ang2(self.conv_ang1(alpha).max(dim=-1, keepdim=False)[0])
        feature = torch.cat([dis_feature, ang_feature], dim=1)
        return feature
