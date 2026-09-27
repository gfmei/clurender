#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/18/2023 11:04 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : common.py
# @Software: PyCharm
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU Usage
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def copy_parameters(model, pretrained, verbose=True):
    model_dict = model.state_dict()
    pretrained_dict = pretrained['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def weights_init(m):
    """
    Xavier normal initialisation for weights and zero bias,
    find especially useful for completion and segmentation Tasks
    """
    classname = m.__class__.__name__
    if (classname.find('Conv1d') != -1) or (classname.find('Conv2d') != -1) or (classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def svm_data(loader, model, is_norm=False):
    feats_list = []
    labels_list = []
    for data in loader:
        data, label = data
        labels = list(map(lambda x: x[0], label.numpy().tolist()))
        data = data.permute(0, 2, 1).cuda()
        with torch.no_grad():
            feats = model.backbone(data)[0]
            if is_norm:
                feats = F.normalize(feats, dim=-1)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_list.append(feat)
        labels_list += labels
    feats_list = np.array(feats_list)
    labels_list = np.array(labels_list)
    return feats_list, labels_list


def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def get_module_device(module):
    return next(module.parameters()).device


def sinkhorn(cost, p, q, epsilon=1e-2, thresh=1e-2, max_iter=100):
    """Log-domain transport with arbitrary leading batch dimensions.

    ``thresh`` measures relative marginal error; zero runs a fixed iteration
    budget. Neither singleton batches nor singleton support axes are squeezed.
    """
    if epsilon <= 0 or max_iter < 1 or thresh < 0:
        raise ValueError("epsilon/max_iter must be positive and thresh nonnegative")
    if cost.shape[-2:] != (p.shape[-1], q.shape[-1]):
        raise ValueError("Transport cost and marginal shapes disagree")
    log_kernel = -cost / epsilon
    log_p, log_q = p.log(), q.log()
    u, v = torch.zeros_like(p), torch.zeros_like(q)
    for _ in range(max_iter):
        u = log_p - torch.logsumexp(log_kernel + v.unsqueeze(-2), dim=-1)
        v = log_q - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
        if thresh > 0:
            log_plan = log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2)
            row_error = (torch.exp(torch.logsumexp(log_plan, -1) - log_p) - 1).abs().amax()
            if row_error.item() < thresh:
                break
    gamma = (log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()
    return gamma, (gamma * cost).sum(dim=(-2, -1))


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def knn(x, k):
    if x.ndim != 3 or not 1 <= k <= x.shape[-1]:
        raise ValueError("Expected B x C x N features and 1 <= k <= N")
    points = x.transpose(1, 2)
    return square_distance(points, points).topk(k=k, dim=-1, largest=False).indices


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two src.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source src, [B, N, C]
        dst: target src, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # A shared translation leaves distances unchanged, but avoids catastrophic
    # cancellation when world coordinates are large compared with separations.
    origin = src[:, :1].detach()
    src, dst = src - origin, dst - origin
    dist = (src.square().sum(-1, keepdim=True)
            + dst.square().sum(-1).unsqueeze(-2)
            - 2 * torch.matmul(src, dst.transpose(-1, -2)))
    return dist.clamp_min(0)


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)  # idx = knn(x[:, :3], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


@torch.no_grad()
def farthest_point_sample(xyz, npoint, is_center=False):
    """Select distinct FPS indices from B x N x C points.

    Center-based initialization chooses the first point furthest from the
    mean; the mean itself is not a sampled point. Ties, including coincident
    points, never cause an index to be selected twice.
    """
    if xyz.ndim != 3 or not 1 <= npoint <= xyz.shape[1]:
        raise ValueError("Expected B x N x C points and 1 <= npoint <= N")
    batch, count, _ = xyz.shape
    centroids = torch.empty(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = xyz.new_full((batch, count), float("inf"))
    batch_indices = torch.arange(batch, device=xyz.device)
    if is_center:
        farthest = (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).argmax(-1)
    else:
        farthest = torch.randint(count, (batch,), device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        distance = torch.minimum(distance, (xyz - centroid).square().sum(-1))
        distance[batch_indices, farthest] = -1
        farthest = distance.argmax(-1)
    return centroids


def index_points(points, idx):
    """
    Input:
        src: input src data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed src data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def transform_points_tsfm(points, viewpoint, inverse=False):
    """Apply OpenCV world-to-camera matrices to row-vector XYZ points.

    Inputs have shapes (..., N, 3) and (..., 4, 4). Positive camera Z
    points forward; X points right and Y points down.
    """
    if points.shape[-1] != 3 or viewpoint.shape[-2:] != (4, 4):
        raise ValueError("Expected XYZ points and 4 x 4 camera matrices")
    rotation = viewpoint[..., :3, :3]
    translation = viewpoint[..., :3, 3].unsqueeze(-2)
    if inverse:
        return torch.linalg.solve(rotation, (points - translation).transpose(-1, -2)).transpose(-1, -2)
    return points @ rotation.transpose(-1, -2) + translation


def points_to_ndc(pts, K, img_dim: List[float], renderer: bool = True):
    """Project OpenCV camera coordinates to PyTorch3D NDC.

    Pixel centers use the integer-index convention (top-left center = (0, 0)).
    The shorter image dimension spans [-1, 1]. Keep signed camera depth so
    rasterizers can reject points behind the camera.
    """
    height, width = img_dim
    if height <= 0 or width <= 0 or K.shape[-2:] != (3, 3):
        raise ValueError("Expected positive image dimensions and 3 x 3 intrinsics")
    projected = pts @ K.transpose(-1, -2)
    depth = pts[..., 2:3]
    denom = projected[..., 2:3]
    denom = torch.where(denom >= 0, denom.clamp_min(1e-8), denom.clamp_max(-1e-8))
    uv = projected[..., :2] / denom
    center = pts.new_tensor([(width - 1) / 2, (height - 1) / 2])
    xy = (uv - center) * (2.0 / min(height, width))
    if renderer:
        xy = -xy
    return torch.cat((xy, depth), dim=-1)


def op_loss(rd_imgs, gt_imgs, lam=0.5):
    """Sum the color/position Wasserstein fitting loss across views."""
    from models.losses import ImageWassersteinLoss
    if len(rd_imgs) != len(gt_imgs) or len(rd_imgs) == 0:
        raise ValueError("Expected the same nonzero number of rendered and target views")
    criterion = ImageWassersteinLoss(position_weight=lam)
    return torch.stack([criterion(x, y) for x, y in zip(rd_imgs, gt_imgs)]).sum()
