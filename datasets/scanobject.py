#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/20/2023 10:34 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : scanobject.py
# @Software: PyCharm
import os

import h5py
from torch.utils.data import Dataset


def load_ScanObjectNN(base_dir, partition):
    data_dir = os.path.join(base_dir, 'main_split')
    h5_name = os.path.join(data_dir, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['datasets'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


class ScanObjectNNSVM(Dataset):
    def __init__(self, root, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(root, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

