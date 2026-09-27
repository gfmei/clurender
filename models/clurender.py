#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2023 4:37 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : clurender.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.common import (square_distance, sinkhorn, get_module_device, feature_transform_regularizer,
                           transform_points_tsfm, points_to_ndc)
from models.renderer import PointRenderer
from models.color import UNetTransformer
from models.losses import BalancedClustering, ImageWassersteinLoss, SlicedImageLoss, round_transport_plan


def ot_assign(x, y, epsilon=1e-3, thresh=1e-3, max_iter=30, dst='fe'):
    device = x.device
    batch_size, dim, num_x = x.shape
    num_y = y.shape[-1]
    # both marginals are fixed with equal weights
    p = torch.empty(batch_size, num_x, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_x)
    q = torch.empty(batch_size, num_y, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_y)
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
    else:
        cost = 2.0 - 2.0 * torch.einsum('bdn,bdm->bnm', x, y)
    gamma, loss = sinkhorn(cost, p, q, epsilon, thresh, max_iter)
    return gamma, loss


def balanced_assign(x, y, dst='fe', max_iter=1000):
    """Balanced pseudo-labels [b, k, n] for points x [b, d, n] and centers y [b, d, k].

    Each point's labels sum to one even when Sinkhorn stops before converging.
    """
    gamma, _ = ot_assign(x, y, max_iter=max_iter, dst=dst)
    batch, num_x, num_y = gamma.shape
    gamma = round_transport_plan(gamma, gamma.new_full((batch, num_x), 1.0 / num_x),
                                 gamma.new_full((batch, num_y), 1.0 / num_y))
    return num_x * gamma.transpose(-1, -2)


def dis_assign(x, y, tau=0.01, dst='eu'):
    """
    :param x:
    :param y: cluster center
    :param tau:
    :param dst:
    :return:
    """
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
        cost_mean = torch.mean(cost, dim=-1, keepdim=True)
        cost = cost_mean - cost
    else:
        cost = 2.0 * torch.einsum('bdn,bdj->bnj', x, y)
    gamma = F.softmax(cost / tau, dim=-1)
    return gamma.transpose(-1, -2), cost


class CONV(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )

    def forward(self, x):
        return self.net(x)


def regular(center, reg=0.0001):
    bs, dim, num = center.shape
    identity = torch.eye(num).to(center).unsqueeze(0)
    loss = reg * torch.abs(torch.einsum('bdm,bdn->bmn', center, center) - identity).mean()
    return loss


class PointCluOT(nn.Module):

    def __init__(self, num_clusters=32, dim=1024, ablation='all'):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        ablation: str Cluster on 'xyz', 'fea' (features), or 'all' (both)
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.dim = dim
        self.ablation = ablation

    def forward(self, feature, xyz):
        bs, dim, num = feature.shape
        # soft-assignment
        log_score = self.conv(feature).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        if self.ablation in ['all', 'xyz']:
            mu_xyz = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
            reg_xyz = 0.001 * regular(mu_xyz)
            with torch.no_grad():
                assign_xyz = balanced_assign(xyz, mu_xyz.detach(), dst='eu')  # [b, k, n]
        else:
            assign_xyz = torch.zeros_like(score).to(xyz)
            reg_xyz = torch.tensor(0.0).to(xyz)
        if self.ablation in ['all', 'fea']:
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2)
            n_mu = F.normalize(mu_fea, dim=1, p=2)
            reg_fea = regular(n_mu)
            with torch.no_grad():
                assign_fea = balanced_assign(n_feature.detach(), n_mu.detach())
        else:
            assign_fea = torch.zeros_like(score).to(xyz)
            reg_fea = torch.tensor(0.0).to(xyz)
        loss_xyz = -torch.mean(torch.sum(assign_xyz.detach() * F.log_softmax(log_score, dim=1), dim=1))
        loss_fea = -torch.mean(torch.sum(assign_fea.detach() * F.log_softmax(log_score, dim=1), dim=1))
        return loss_xyz + loss_fea + reg_fea + reg_xyz


class PointCluDS(nn.Module):
    def __init__(self, num_clusters=32, dim=1024, ablation='all'):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        ablation: str Cluster on 'xyz', 'fea' (features), or 'all' (both)
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.dim = dim
        self.ablation = ablation

    def forward(self, feature, xyz):
        bs, dim, num = feature.shape
        # soft-assignment
        log_score = self.conv(feature).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        if self.ablation in ['all', 'xyz']:
            mu_xyz = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
            reg_xyz = 0.001 * regular(mu_xyz)
            with torch.no_grad():
                assign_xyz, dis = dis_assign(xyz, mu_xyz.detach(), dst='eu')
        else:
            assign_xyz = torch.zeros_like(score).to(xyz)
            reg_xyz = torch.tensor(0.0).to(xyz)
        if self.ablation in ['all', 'fea']:
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2)
            n_mu = F.normalize(mu_fea, dim=1, p=2)
            reg_fea = regular(n_mu)
            with torch.no_grad():
                assign_fea, dis = dis_assign(n_feature.detach(), n_mu.detach())
        else:
            assign_fea = torch.zeros_like(score).to(xyz)
            reg_fea = torch.tensor(0.0).to(xyz)
        loss_xyz = -torch.mean(torch.sum(assign_xyz.detach() * F.log_softmax(log_score, dim=1), dim=1))
        loss_fea = -torch.mean(torch.sum(assign_fea.detach() * F.log_softmax(log_score, dim=1), dim=1))
        return loss_xyz + loss_fea + reg_fea + reg_xyz


class ClusterNet(nn.Module):
    def __init__(self,
                 backbone,
                 dim=1024,
                 num_clus=64,
                 num_clus1=None,
                 ablation='all',
                 c_type='ot'):
        super().__init__()
        self.backbone = backbone
        if c_type == 'ot':
            self.cluster = PointCluOT(num_clusters=num_clus, dim=dim, ablation=ablation)
        else:
            self.cluster = PointCluDS(num_clusters=num_clus, dim=dim, ablation=ablation)
        self.num_clus1 = num_clus1
        device = get_module_device(backbone)
        self.to(device)

    def forward(self, x, return_embedding=False):
        """
        :param x: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding:
            return self.backbone(x)[0]
        out = self.backbone(x)
        trans_loss = x.new_zeros(())
        if len(out) == 2:
            feature, wise = out
        else:
            feature, wise, trans = out
            if trans is not None:
                trans_loss = 0.001 * feature_transform_regularizer(trans)
        loss_rq = self.cluster(wise, x[:, :3])

        return loss_rq, trans_loss


class CluRender(nn.Module):
    """Joint clustering and multi-view neural rendering pretraining.

    ``points``: B x 3 x N, ``images``: B x V x 3 x H x W,
    ``tsfms``: B x V x 4 x 4 world-to-camera OpenCV transforms,
    ``K``: B x V x 3 x 3 (or shared B x 3 x 3) pixel intrinsics.
    Lists of B-sized views are also accepted for the original interface.
    The total loss is clustering + render_weight * rendering + transform;
    ``return_details`` reports the unweighted rendering term.
    """

    def __init__(self, backbone, dim=1024, num_clus=64, render_cfg=None,
                 render_dim=3, c_type='ot', image_loss='sinkhorn',
                 ot_backend='auto', image_blur=0.01, num_projections=128,
                 sinkhorn_iterations=2000, epsilon=0.001, orthogonal_weight=0.01,
                 color_dims=(512, 256, 128, 128, 64, 32),
                 round_assignments=True, checkpoint_rendering=False,
                 sinkhorn_tolerance=0.01, render_weight=1.0):
        super().__init__()
        if render_dim != 3 or c_type != 'ot':
            raise ValueError("CluRender pretraining requires RGB output and OT clustering")
        if len(color_dims) != 6:
            raise ValueError("color_dims must contain six dimensions")
        if not math.isfinite(render_weight) or render_weight < 0:
            raise ValueError("render_weight must be finite and nonnegative")
        self.backbone = backbone
        self.cluster = BalancedClustering(dim, num_clus, epsilon, sinkhorn_iterations,
                                          orthogonal_weight, round_assignments, sinkhorn_tolerance)
        self.render_weight = render_weight
        self.checkpoint_rendering = checkpoint_rendering
        self.color = UNetTransformer(dim, 3, d_dims=color_dims[:3], u_dims=color_dims[3:])
        self.render = PointRenderer(render_cfg)
        if image_loss == 'sinkhorn':
            self.fitting = ImageWassersteinLoss(blur=image_blur, backend=ot_backend)
        elif image_loss == 'sliced':
            self.fitting = SlicedImageLoss(num_projections)
        else:
            raise ValueError("image_loss must be 'sinkhorn' or 'sliced'")
        self.to(get_module_device(backbone))

    def _render_view(self, ndc, colors):
        rendered = self.render(ndc, colors, return_raster=False)
        return rendered['feats'], rendered['valid_rays'].detach().mean()

    def forward(self, points, images=None, tsfms=None, K=None,
                return_embedding=False, return_details=False):
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError("points must have shape (B, 3, N)")
        out = self.backbone(points)
        if return_embedding:
            return out[0]
        if images is None or tsfms is None or K is None:
            raise ValueError("Joint pretraining requires paired images, transforms, and intrinsics")
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=1)
        if isinstance(tsfms, (list, tuple)):
            tsfms = torch.stack(tsfms, dim=1)
        if isinstance(K, (list, tuple)):
            K = torch.stack(K, dim=1)
        batch = points.shape[0]
        if images.ndim != 5 or images.shape[0] != batch or images.shape[2] != 3:
            raise ValueError("images must have shape (B, V, 3, H, W)")
        views = images.shape[1]
        if views == 0 or tsfms.shape != (batch, views, 4, 4):
            raise ValueError("Provide one 4 x 4 world-to-camera transform per image")
        if K.shape == (batch, 3, 3):
            K = K.unsqueeze(1).expand(-1, views, -1, -1)
        if K.shape != (batch, views, 3, 3):
            raise ValueError("Intrinsics must be (B, 3, 3) or (B, V, 3, 3)")
        height, width = images.shape[-2:]
        if (height, width) != self.render.image_size:
            raise ValueError("Target image dimensions must match render_size")
        wise = out[1]
        trans_loss = points.new_zeros(())
        if len(out) > 2 and out[2] is not None:
            trans_loss = 0.001 * feature_transform_regularizer(out[2])
        xyz = points.transpose(1, 2)
        colors = self.color(xyz, wise.transpose(1, 2)).sigmoid().transpose(1, 2)
        view_losses = []
        coverage = []
        for view in range(views):
            camera_points = transform_points_tsfm(xyz, tsfms[:, view])
            ndc = points_to_ndc(camera_points, K[:, view], [height, width])
            if self.checkpoint_rendering and torch.is_grad_enabled() and colors.requires_grad:
                image, visible = checkpoint(self._render_view, ndc, colors, use_reentrant=False)
            else:
                image, visible = self._render_view(ndc, colors)
            view_losses.append(self.fitting(image, images[:, view]))
            coverage.append(visible)
        loss_clu, clustering_details = self.cluster(wise, points, return_details=True)
        loss_render = torch.stack(view_losses).sum()
        total = loss_clu + self.render_weight * loss_render + trans_loss
        if return_details:
            return {"loss": total, "clustering": loss_clu, "rendering": loss_render,
                    "transform": trans_loss, "coverage": torch.stack(coverage).mean(),
                    "assignment_error": clustering_details["unrounded_marginal_error"],
                    "batch_size": points.new_tensor(batch)}
        return total
