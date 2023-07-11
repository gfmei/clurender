#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2023 4:37 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : clurender.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from models.common import (square_distance, sinkhorn, get_module_device, feature_transform_regularizer,
                           transform_points_tsfm, points_to_ndc, farthest_point_sample, index_points)
from models.renderer import PointRenderer


class TransformerLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TransformerLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.query = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.key = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.value = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.fc = nn.Conv1d(dim_out, dim_out, kernel_size=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        attn_output = torch.bmm(v, attn_weights.permute(0, 2, 1))
        attn_output = self.fc(attn_output)

        return attn_output + x


class TransformerDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, num_points, is_center=True):
        super(TransformerDownSampling, self).__init__()
        self.num_points = num_points
        self.is_center = is_center
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1),
            nn.InstanceNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, points, features):
        features = self.conv(features.transpose(-1, -2)).transpose(-1, -2)
        idx = farthest_point_sample(points, self.num_points, is_center=self.is_center)

        sampled_points = index_points(points, idx)
        sampled_features = index_points(features, idx)

        return sampled_points, sampled_features


class TransformerUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransformerUpSampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1),
            nn.InstanceNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, xyz1, xyz2, features1, features2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_feastures = features2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feastures = torch.sum(index_points(features2, idx) * weight.view(B, N, 3, 1), dim=2)

        if features1 is not None:
            new_featuress = torch.cat([features1, interpolated_feastures], dim=-1)
        else:
            new_featuress = interpolated_feastures
        new_featuress = new_featuress.permute(0, 2, 1)
        new_featuress = self.conv(new_featuress)

        return new_featuress.permute(0, 2, 1)


class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_samples_list, d_dims, u_dims, num_heads, is_center=True):
        super(UNetTransformer, self).__init__()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.is_center = is_center
        self.num_stages = len(num_samples_list)  # Number of downsampling stages

        # Downsampling layers
        self.conv = nn.Conv1d(in_channels, d_dims[0], kernel_size=1)
        for i in range(self.num_stages):
            self.downsampling.append(
                TransformerDownSampling(in_channels, d_dims[i], num_samples_list[i], self.is_center))
            in_channels = d_dims[i]
            self.encoder_layers.append(TransformerEncoderLayer(d_dims[i], num_heads))

        # Upsampling layers
        for i in range(self.num_stages):
            in_channels = d_dims[self.num_stages - i - 1] + d_dims[self.num_stages - i - 2]
            self.upsampling.append(TransformerUpSampling(in_channels, u_dims[i]))
            self.decoder_layers.append(TransformerEncoderLayer(u_dims[i], num_heads))
        self.final_conv = nn.Conv1d(u_dims[-1], out_channels, kernel_size=1)

    def forward(self, points, features):
        skip_connections = []
        # Downsample
        sampled_points = points
        ds_features = self.conv(features.transpose(-1, -2)).transpose(-1, -2)
        for i in range(self.num_stages):
            # sample points and features
            sampled_points, ds_features = self.downsampling[i](sampled_points, ds_features)
            skip_connections.append((sampled_points, ds_features))
            ds_features = self.encoder_layers[i](ds_features)

        # Upsample
        for i in range(self.num_stages):
            xyz1, features1 = skip_connections[self.num_stages - i - 1]
            xyz2, features2 = skip_connections[self.num_stages - i - 2]
            features = self.upsampling[i](xyz1, xyz2, features1, features2)
            features = self.decoder_layers[i](features)

        # Final convolution
        output = self.final_conv(features.transpose(-1, -2))

        return output


def ot_assign(x, y, epsilon=1e-3, thresh=1e-3, max_iter=30, dst='fe'):
    device = x.device
    batch_size, dim, num_x = x.shape
    num_y = y.shape[-1]
    # both marginals are fixed with equal weights
    p = torch.empty(batch_size, num_x, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
    q = torch.empty(batch_size, num_y, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
    else:
        cost = 2.0 - 2.0 * torch.einsum('bdn,bdm->bnm', x, y)
    gamma, loss = sinkhorn(cost, p, q, epsilon, thresh, max_iter)
    return gamma, loss


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


class SlicedWassersteinDistance(nn.Module):
    def __init__(self, num_projections=1000):
        super(SlicedWassersteinDistance, self).__init__()
        self.num_projections = num_projections

    def forward(self, x, y):
        assert x.shape == y.shape
        b, w, h, c = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, c, -1)  # Flatten spatial dimensions
        y = y.permute(0, 3, 1, 2).reshape(b, c, -1)

        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        grid = torch.stack((grid_x, grid_y), 0).to(x.device)  # 2, W, H
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1).reshape(b, 2, -1)  # Add coordinates as channels

        x = torch.cat((x, grid), 1)  # Add the coordinates to the feature dimension
        y = torch.cat((y, grid), 1)

        projections = torch.randn((self.num_projections, 5)).to(x.device)

        x_projections = (x.unsqueeze(1) * projections.view(1, self.num_projections, 5, 1)).sum(dim=-2)
        y_projections = (y.unsqueeze(1) * projections.view(1, self.num_projections, 5, 1)).sum(dim=-2)

        swd = (x_projections.sort(dim=-1)[0] - y_projections.sort(dim=-1)[0]).pow(2).mean().sqrt()
        return swd


def renderer_loss(rendered_imgs, gt_imgs, num_projections=1000):
    swd = SlicedWassersteinDistance(num_projections)
    loss = swd(rendered_imgs, gt_imgs)
    return loss


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
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
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
                assign_xyz, dis = ot_assign(xyz, mu_xyz.detach(), max_iter=25, dst='eu')
                assign_xyz = num * assign_xyz.transpose(-1, -2)  # [b, k, n]
        else:
            assign_xyz = torch.zeros_like(score).to(xyz)
            reg_xyz = torch.tensor(0.0).to(xyz)
        if self.ablation in ['all', 'fea']:
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2)
            n_mu = F.normalize(mu_fea, dim=1, p=2)
            reg_fea = regular(n_mu)
            with torch.no_grad():
                assign_fea, dis = ot_assign(n_feature.detach(), n_mu.detach(), max_iter=25)
                assign_fea = num * assign_fea.transpose(-1, -2)
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
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
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
            return self.backbone(x, True)
        out = self.backbone(x)
        trans_loss = torch.tensor(0.0, requires_grad=True)
        if len(out) == 2:
            feature, wise = out
        else:
            feature, wise, trans = out
            if trans is not None:
                trans_loss = 0.001 * feature_transform_regularizer(trans)
        loss_rq = self.cluster(wise, x)

        return loss_rq, trans_loss


class CluRender(nn.Module):
    def __init__(self,
                 backbone,
                 dim=1024,
                 num_clus=64,
                 render_cfg=None,
                 render_dim=3,
                 c_type='ot'):
        super().__init__()
        self.backbone = backbone
        self.cluster = PointCluOT(num_clusters=num_clus, dim=dim)
        self.color = CONV(in_size=dim, out_size=render_dim, hidden_size=dim // 2, used='proj')
        self.render = PointRenderer(render_cfg)
        self.c_type = c_type
        device = get_module_device(backbone)
        self.to(device)

    def forward(self, points, images=None, tsfms=None, K=None, return_embedding=False):
        """
        :param images:
        :param tsfms:
        :param points: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding:
            return self.backbone(points, True)
        out = self.backbone(points)
        trans_loss = torch.tensor(0.0, requires_grad=True)
        if len(out) == 2:
            feature, wise = out
        else:
            feature, wise, trans = out
            if trans is not None:
                trans_loss = 0.001 * feature_transform_regularizer(trans)
        render_loss = []
        pcd_lists = [transform_points_tsfm(points.transpose(1, 2), tsfm) for tsfm in tsfms]
        colors = self.color(wise)
        B, _, H, W = images[0].shape
        for i in range(len(tsfms)):
            pcd_i = points_to_ndc(pcd_lists[i], K, [H, W])
            render_loss_i = render_loss(self.render(pcd_i, colors), images[i])
            render_loss.append(render_loss_i)
        loss_clu = self.cluster(wise, points)
        trans_loss += loss_clu + torch.tensor(render_loss).sum()

        return trans_loss


if __name__ == '__main__':
    # Example usage
    # in_channels = 32
    # out_channels = 32
    # N = 1024  # Number of input points
    # num_samples_list = [512, 256, 128]
    # points = torch.randn(2, N, 3)  # Input point coordinates
    # features = torch.randn(2, N, in_channels)  # Input point features
    # model = UNetTransformer(in_channels, out_channels, num_samples_list, [32, 64, 128], [128, 64, 32], 2)
    # output = model(points, features)
    # Example usage
    batch_size = 4
    width = 128
    height = 128

    # Generate random rendered and ground truth images
    rendered_images = torch.randn(batch_size, width, height, 3)
    ground_truth_images = torch.randn(batch_size, width, height, 3)

    # Calculate image loss
    loss = renderer_loss(rendered_images, ground_truth_images)

    print(loss)

