#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/20/2023 9:41 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : lib_utils.py
# @Software: PyCharm
import datetime
import glob
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
# Basic libs
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.nn.init as initer
from matplotlib import pyplot as plt
from plyfile import PlyData
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass


def bn_momentum_adjust(m, momentum):
    if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.BatchNorm1d):
        m.momentum = momentum


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    target[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def calc_victim_value(class_value, label, victim_class):
    values = []
    for lbl in victim_class:
        if label is None or (label == lbl).any():
            values.append(class_value[lbl])
    return np.mean(values)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split('.')) > 1:
            recursive_set(
                getattr(cur_module, name[:name.find('.')]), name[name.find('.')+1:], module)
        else:
            setattr(cur_module, name, module)
    from sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, \
        SynchronizedBatchNorm3d
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(model, name, SynchronizedBatchNorm1d(
                m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(model, name, SynchronizedBatchNorm2d(
                m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(model, name, SynchronizedBatchNorm3d(
                m.num_features, m.eps, m.momentum, m.affine))


def lbl2rgb(label, names):
    """Convert label to rgb colors.
    label: [N]
    """
    from config import NAME2COLOR
    if len(names) == 13:
        colors = NAME2COLOR['S3DIS']
    else:
        colors = NAME2COLOR['ScanNet']
    rgb = np.zeros((label.shape[0], 3))
    uni_lbl = np.unique(label).astype(np.uint8)
    for lbl in uni_lbl:
        mask = (label == lbl)
        rgb[mask] = np.tile(np.array(
            colors[names[lbl]])[None, :], (mask.sum(), 1))
    return rgb


def convert2vis(xyz, label, names):
    """Assign color to each point according to label."""
    rgb = lbl2rgb(label, names) * 255.
    data = np.concatenate([xyz, rgb], axis=1)
    return data


def proc_pert(points, gt, pred, folder,
              names, part=False, ignore_label=255):
    """Process and save files for visulization in perturbation attack."""
    check_makedirs(folder)
    lbl2cls = {i: names[i] for i in range(len(names))}

    np.savetxt(os.path.join(folder, 'all_points.txt'), points, delimiter=';')
    gt_seg = convert2vis(points[gt != ignore_label, :3],
                         gt[gt != ignore_label], names)
    pred_seg = convert2vis(points[gt != ignore_label, :3],
                           pred[gt != ignore_label], names)
    np.savetxt(os.path.join(folder, 'gt.txt'),
               gt_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'pred.txt'),
               pred_seg, delimiter=';')
    if part:
        uni_lbl = np.unique(gt[gt != ignore_label]).astype(np.uint8)
        for lbl in uni_lbl:
            lbl = int(lbl)
            mask = (gt == lbl)
            sel_points = points[mask]
            mask = (gt[gt != ignore_label] == lbl)
            sel_seg = pred_seg[mask]
            np.savetxt(
                os.path.join(folder, '{}_{}_points.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_points, delimiter=';')
            np.savetxt(
                os.path.join(folder, '{}_{}_pred.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_seg, delimiter=';')


def proc_add(points, noise, gt, pred, noise_pred, folder,
             names, part=False, ignore_label=255):
    """Process and save files for visulization in adding attack."""
    check_makedirs(folder)
    lbl2cls = {i: names[i] for i in range(len(names))}

    np.savetxt(os.path.join(folder, 'all_points.txt'), points, delimiter=';')
    np.savetxt(os.path.join(folder, 'noise_points.txt'), noise, delimiter=';')
    gt_seg = convert2vis(points[gt != ignore_label, :3],
                         gt[gt != ignore_label], names)
    pred_seg = convert2vis(points[gt != ignore_label, :3],
                           pred[gt != ignore_label], names)
    noise_seg = convert2vis(noise[:, :3], noise_pred, names)
    np.savetxt(os.path.join(folder, 'gt.txt'),
               gt_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'pred.txt'),
               pred_seg, delimiter=';')
    np.savetxt(os.path.join(folder, 'noise_pred.txt'),
               noise_seg, delimiter=';')
    if part:
        uni_lbl = np.unique(gt[gt != ignore_label]).astype(np.uint8)
        for lbl in uni_lbl:
            lbl = int(lbl)
            mask = (gt == lbl)
            sel_points = points[mask]
            mask = (gt[gt != ignore_label] == lbl)
            sel_seg = pred_seg[mask]
            np.savetxt(
                os.path.join(folder, '{}_{}_points.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_points, delimiter=';')
            np.savetxt(
                os.path.join(folder, '{}_{}_pred.txt'.format(
                    lbl, lbl2cls[lbl])),
                sel_seg, delimiter=';')


def save_vis(pred_root, save_root, data_root):
    from config import CLASS_NAMES
    if 'S3DIS' in data_root:  # save Area5 data
        names = CLASS_NAMES['S3DIS']['other']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_5.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_5.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        all_rooms = sorted(os.listdir(data_root))
        all_rooms = [
            room for room in all_rooms if 'Area_5' in room
        ]
        assert len(gt_save) == len(all_rooms)
        check_makedirs(save_root)
        for i, room in enumerate(all_rooms):
            points = np.load(os.path.join(data_root, room))[:, :6]
            folder = os.path.join(save_root, room[:-4])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)
    elif 'ScanNet' in data_root:  # save val set data
        names = CLASS_NAMES['ScanNet']['other']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_val.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_val.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_file = os.path.join(
            data_root, 'scannet_val_rgb21c_pointid.pickle')
        file_pickle = open(data_file, 'rb')
        xyz_all = pickle.load(file_pickle)
        file_pickle.close()
        assert len(xyz_all) == len(gt_save)
        with open(os.path.join(
                data_root, 'meta_data/scannetv2_val.txt')) as fl:
            scene_id = fl.read().splitlines()
        assert len(scene_id) == len(gt_save)
        check_makedirs(save_root)
        for i in range(len(gt_save)):
            points = xyz_all[i][:, :6]
            folder = os.path.join(save_root, scene_id[i])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)


def save_vis_mink(pred_root, save_root, data_root):
    from config import CLASS_NAMES

    def load_data(file_name):
        plydata = PlyData.read(file_name)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']],
                          dtype=np.float32).T
        colors = np.array([data['red'], data['green'],
                           data['blue']], dtype=np.float32).T
        return np.concatenate([coords, colors], axis=1)

    if 'S3DIS' in data_root:  # save Area5 data
        names = CLASS_NAMES['S3DIS']['mink']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_5.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_5.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_root = os.path.join(data_root, 'Area_5')
        all_rooms = sorted(os.listdir(data_root))
        assert len(all_rooms) == len(gt_save)
        check_makedirs(save_root)

        for i, room in enumerate(all_rooms):
            data = os.path.join(data_root, room)
            points = load_data(data)
            folder = os.path.join(
                save_root, 'Area_5_{}'.format(room[:-4]))
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)
    elif 'ScanNet' in data_root:  # save val set
        names = CLASS_NAMES['ScanNet']['mink']
        gt_save = load_pickle(
            os.path.join(pred_root, 'gt_val.pickle'))['gt']
        pred_save = load_pickle(
            os.path.join(pred_root, 'pred_val.pickle'))['pred']
        assert len(gt_save) == len(pred_save)
        data_root = os.path.join(data_root, 'train')
        with open(os.path.join(
                data_root, 'scannetv2_val.txt'), 'r') as f:
            all_rooms = f.readlines()
        all_rooms = [room[:-1] for room in all_rooms]
        assert len(all_rooms) == len(gt_save)
        check_makedirs(save_root)

        for i, room in enumerate(all_rooms):
            data = os.path.join(data_root, room)
            points = load_data(data)
            folder = os.path.join(save_root, room[:-4])
            check_makedirs(folder)
            proc_pert(points, gt_save[i], pred_save[i],
                      folder, names, part=True)


def save_vis_from_pickle(pkl_root, save_root=None, room_idx=52,
                         room_name='scene0354_00'):
    names = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
        'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
        'curtain', 'refrigerator', 'showercurtain', 'toilet', 'sink',
        'bathtub', 'otherfurniture'
    ]
    data = load_pickle(pkl_root)
    points = data['data'][room_idx]
    pred = data['pred'][room_idx]
    gt = data['gt'][room_idx]
    if save_root is None:
        save_root = os.path.dirname(pkl_root)
    save_folder = os.path.join(save_root, room_name)
    proc_pert(points, gt, pred, save_folder, names, part=True)


def save_pickle(filename, dict_data):
    with open(filename, 'wb') as handle:
        pickle.dump(dict_data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def load_s3dis_instance(folder, name2cls, load_name=['chair']):
    """Load S3DIS room in a Inst Seg format.
    Get each instance separately.
    If load_name is None or [], return all instances.
    Returns a list of [np.array of [N, 6], label]
    """
    cls2name = {name2cls[name]: name for name in name2cls.keys()}
    anno_path = os.path.join(folder, 'Annotations')
    points_list = []
    labels_list = []
    idx = 0
    files = glob.glob(os.path.join(anno_path, '*.txt'))
    files.sort()

    for f in files:
        cls = os.path.basename(f).split('_')[0]
        if cls not in name2cls.keys():
            cls = 'clutter'
        points = np.loadtxt(f)  # [N, 6]
        num = points.shape[0]
        points_list.append(points)
        labels_list.append((idx, idx + num, name2cls[cls]))
        idx += num

    # normalize points coords by minus min
    data = np.concatenate(points_list, 0)
    xyz_min = np.amin(data, axis=0)[0:3]
    data[:, 0:3] -= xyz_min

    # rearrange to separate instances
    if load_name is None or not load_name:
        load_name = list(name2cls.keys())
    instances = [
        [data[pair[0]:pair[1]], pair[2]] for pair in labels_list if
        cls2name[pair[2]] in load_name
    ]
    return instances


def cal_loss(pred, gold, smoothing=False, ignore_index=255):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(
            pred, gold, reduction='mean',
            ignore_index=ignore_index)

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:

        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):
    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    points = np.random.rand(10, 3)
    write_ply('example1.ply', points, ['x', 'y', 'z'])
    values = np.random.randint(2, size=10)
    write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

            # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

        # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def get_colored_point_cloud_tsne(points, feature, name):
    import open3d as o3d
    tsne_results = embed_tsne(feature)
    color = get_color_map(tsne_results)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(name, pcd)
    return pcd


def make_point_cloud(points, name, color=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(name, pcd)
    return pcd


def get_colored_point_cloud_pca(xyz_list, features, name, num):
    """N x D"""
    import open3d as o3d
    pca = PCA(n_components=3)
    pca_gf = pca.fit_transform(features)
    pca_gf = (pca_gf + np.abs(pca_gf.min(0))) / pca_gf.ptp(0)
    length = len(xyz_list)
    for i in range(length):
        points = xyz_list[i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(pca_gf[i * num:(i + 1) * num])
        o3d.io.write_point_cloud(name + f'{i}.ply', pcd)


def save_pcd(xyz, name, color=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(name, pcd)


def get_colored_point_cloud_pca_sep(xyz, feature, name):
    """N x D"""
    import open3d as o3d
    pca = PCA(n_components=3)
    pca_gf = pca.fit_transform(feature)
    pca_gf = (pca_gf + np.abs(pca_gf.min(0))) / pca_gf.ptp(0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(pca_gf)
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def embed_tsne(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class TrainLogger:

    def __init__(self, args, name='model', subfold='cls', filename='train_log', cls2name=None):
        self.step = 1
        self.epoch = 1
        self.args = args
        self.name = name
        self.sf = subfold
        self.mkdir()
        self.setup(filename=filename)
        self.epoch_init()
        self.save_model = False
        self.cls2name = cls2name
        self.best_instance_acc, self.best_class_acc, self.best_miou = 0., 0., 0.
        self.best_instance_epoch, self.best_class_epoch, self.best_miou_epoch = 0, 0, 0
        self.savepath = str(self.checkpoints_dir) + '/best_model.pth'

    def setup(self, filename='train_log'):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, filename + '.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # ref: https://stackoverflow.com/a/53496263/12525201
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # logging.getLogger('').addHandler(console) # this is root logger
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)
        self.logger.removeHandler(console)

    def mkdir(self):
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./log/')
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath(self.sf)
        experiment_dir.mkdir(exist_ok=True)

        if self.args.log_dir is None:
            self.experiment_dir = experiment_dir.joinpath(timestr)
        else:
            self.experiment_dir = experiment_dir.joinpath(self.args.log_dir)

        self.experiment_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.log_dir = self.experiment_dir.joinpath('logs/')
        self.log_dir.mkdir(exist_ok=True)

    # @property.setter
    def epoch_init(self, training=True):
        self.loss, self.count, self.pred, self.gt = 0., 0., [], []
        if training:
            self.logger.info('Epoch %d/%d:' % (self.epoch, self.args.epoch))

    def step_update(self, pred, gt, loss, training=True):
        if training:
            self.step += 1  # Use TensorFlow way to count training steps
        self.gt.append(gt)
        self.pred.append(pred)
        batch_size = len(pred)
        self.count += batch_size
        self.loss += loss * batch_size

    def epoch_update(self, training=True, mode='cls'):
        self.save_model = False
        self.gt = np.concatenate(self.gt)
        self.pred = np.concatenate(self.pred)

        instance_acc = metrics.accuracy_score(self.gt, self.pred)
        if instance_acc > self.best_instance_acc and not training:
            self.save_model = True if mode == 'cls' else False
            self.best_instance_acc = instance_acc
            self.best_instance_epoch = self.epoch

        if mode == 'cls':
            class_acc = metrics.balanced_accuracy_score(self.gt, self.pred)
            if class_acc > self.best_class_acc and not training:
                self.best_class_epoch = self.epoch
                self.best_class_acc = class_acc
            return instance_acc, class_acc
        elif mode == 'semseg':
            miou = self.calculate_IoU().mean()
            if miou > self.best_miou and not training:
                self.best_miou_epoch = self.epoch
                self.save_model = True
                self.best_miou = miou
            return instance_acc, miou
        else:
            raise ValueError('Mode is not Supported by TrainLogger')

    def epoch_summary(self, writer=None, training=True, mode='cls'):
        criteria = 'Class Accuracy' if mode == 'cls' else 'mIoU'
        instance_acc, class_acc = self.epoch_update(training=training, mode=mode)
        if training:
            if writer is not None:
                writer.add_scalar('Train Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Train %s' % criteria, class_acc, self.step)
            self.logger.info('Train Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Train %s: %.3f' % (criteria, class_acc))
        else:
            if writer is not None:
                writer.add_scalar('Test Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Test %s' % criteria, class_acc, self.step)
            self.logger.info('Test Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Test %s: %.3f' % (criteria, class_acc))
            self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
                self.best_instance_acc, self.best_instance_epoch))
            if self.best_class_acc > .1:
                self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                    self.best_class_acc, self.best_class_epoch))
            if self.best_miou > .1:
                self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                    self.best_miou, self.best_miou_epoch))

        self.epoch += 1 if not training else 0
        if self.save_model:
            self.logger.info('Saving the Model Params to %s' % self.savepath)

    def calculate_IoU(self):
        num_class = len(self.cls2name)
        Intersection = np.zeros(num_class)
        Union = Intersection.copy()
        # self.pred -> numpy.ndarray (total predictions, )

        for sem_idx in range(num_class):
            Intersection[sem_idx] = np.sum(np.logical_and(self.pred == sem_idx, self.gt == sem_idx))
            Union[sem_idx] = np.sum(np.logical_or(self.pred == sem_idx, self.gt == sem_idx))
        return Intersection / Union

    def train_summary(self, mode='cls'):
        self.logger.info('\n\nEnd of Training...')
        self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
            self.best_instance_acc, self.best_instance_epoch))
        if mode == 'cls':
            self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                self.best_class_acc, self.best_class_epoch))
        elif mode == 'semseg':
            self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                self.best_miou, self.best_miou_epoch))

    def update_from_checkpoints(self, checkpoint):
        self.logger.info('Use Pre-Trained Weights')
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_instance_epoch, self.best_instance_acc = checkpoint['epoch'], checkpoint['instance_acc']
        self.best_class_epoch, self.best_class_acc = checkpoint['best_class_epoch'], checkpoint['best_class_acc']
        self.logger.info('Best Class Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_class_epoch))
        self.logger.info(
            'Best Instance Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_instance_epoch))
