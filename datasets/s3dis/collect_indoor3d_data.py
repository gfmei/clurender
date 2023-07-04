#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2023 2:05 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : collect_indoor3d_data.py
# @Software: PyCharm
import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
SYNSET_DICT_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath('../../'))
import indoor3d_util
# Constants
ROOT_DIR = '/data/disk1/data'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'Stanford3dDataset')

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, '../metas/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(DATA_PATH, 'stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# revise_file = os.path.join(DATA_PATH, "Area_5/hallway_6/Annotations/ceiling_1.txt")
# with open(revise_file, "r") as f:
#     data = f.read()
#     data = data[:5545347] + ' ' + data[5545348:]
#     f.close()
# with open(revise_file, "w") as f:
#     f.write(data)
#     f.close()

for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
        indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
