#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/25/2023 2:09 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : main_semseg_s3dis.py
# @Software: PyCharm
import argparse
import os

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plyfile import PlyData, PlyElement
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.all_data import S3DIS
from lib.lib_utils import cal_loss, IOStream
from models.clurender import ClusterNet
from models.dgcnn import DGCNNSegS3dis


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_semseg_s3dis.py checkpoints' + '/' + args.exp_name + '/' + 'main_semseg_s3dis.py.backup')


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def visualization(root, visu, visu_format, test_choice, data, seg, clu, crd, visual_file_index, semseg_colors):
    room_gt, room_soft, room_crd = [], [], []
    visual_warning = True
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB_clu = []
        RGB_gt = []
        RGB_crd = []
        skip = False
        with open(os.path.join(root, "indoor3d_sem_seg_hdf5_data_test/room_filelist.txt")) as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index + 1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != \
                        roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False
            else:
                skip = True
        elif test_choice != 'all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists(
                    'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + str(test_area)
                    + '/' + roomname):
                os.makedirs(
                    'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + str(test_area)
                    + '/' + roomname)
            room_path = os.path.join(root, 'indoor3d_sem_seg_hdf5_data_test/raw_data3d', 'Area_' + str(test_area)
                                     + '/' + roomname + '(' + str(visual_file_index) + ').txt')
            data = np.loadtxt(room_path)
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB_clu.append(semseg_colors[int(clu[i][j])])
                RGB_crd.append(semseg_colors[int(crd[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:, [1, 2, 0]]
            xyzRGB_clu = np.concatenate((data, np.array(RGB_clu)), axis=1)
            xyzRGB_crd = np.concatenate((data, np.array(RGB_crd)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_gt.append(seg[i].cpu().numpy())
            room_soft.append(clu[i].cpu().numpy())
            room_crd.append(crd[i].cpu().numpy())
            ply_path = 'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname
            if not os.path.exists(ply_path):
                os.makedirs(ply_path, exist_ok=False)
            with open(os.path.join(ply_path, roomname + '_soft.txt'), "a") as f:
                np.savetxt(f, xyzRGB_clu, fmt='%s', delimiter=' ')
            with open(ply_path + '/' + roomname + '_crd.txt', "a") as f_gt:
                np.savetxt(f_gt, xyzRGB_crd, fmt='%s', delimiter=' ')
            with open(ply_path + '/' + roomname + '_gt.txt', "a") as f_gt:
                np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ')

            if roomname != roomname_next:
                mIoU_soft = np.nanmean(calculate_sem_IoU(np.array(room_soft), np.array(room_gt)))
                mIoU_soft = str(round(mIoU_soft, 4))
                mIoU_crd = np.nanmean(calculate_sem_IoU(np.array(room_crd), np.array(room_gt)))
                mIoU_crd = str(round(mIoU_crd, 4))
                room_soft = []
                room_crd = []
                room_gt = []
                if visu_format == 'ply':
                    filepath_soft = ply_path + '/' + roomname + '_soft_' + mIoU_soft + '.ply'
                    filepath_crd = ply_path + '/' + roomname + '_crd_' + mIoU_crd + '.ply'
                    filepath_gt = ply_path + '/' + roomname + '_gt.ply'
                    xyzRGB_clu = np.loadtxt(ply_path + '/' + roomname + '_soft.txt')
                    xyzRGB_crd = np.loadtxt(ply_path + '/' + roomname + '_crd.txt')
                    xyzRGB_gt = np.loadtxt(ply_path + '/' + roomname + '_gt.txt')
                    xyzRGB_clu = [(xyzRGB_clu[i, 0], xyzRGB_clu[i, 1], xyzRGB_clu[i, 2], xyzRGB_clu[i, 3],
                                    xyzRGB_clu[i, 4], xyzRGB_clu[i, 5]) for i
                                   in range(xyzRGB_clu.shape[0])]
                    xyzRGB_crd = [(xyzRGB_crd[i, 0], xyzRGB_crd[i, 1], xyzRGB_crd[i, 2], xyzRGB_crd[i, 3],
                                   xyzRGB_crd[i, 4], xyzRGB_crd[i, 5]) for i
                                  in range(xyzRGB_crd.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4],
                                  xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(
                        xyzRGB_clu, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                            ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_soft)
                    print('PLY visualization file saved in', filepath_soft)
                    vertex = PlyElement.describe(np.array(
                        xyzRGB_crd, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                           ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_crd)
                    print('PLY visualization file saved in', filepath_crd)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt,
                                                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system(
                        'rm -rf ' + 'checkpoints/' + args.exp_name + '/visualization/area_'
                        + test_area + '/' + roomname + '/*.txt')
                else:
                    filename_crd_mIoU = 'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' \
                                        + test_area + '/' + roomname + '/' + roomname + '_crd_' + mIoU_crd + '.txt'
                    filename_gt = 'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + \
                                  test_area + '/' + roomname + '/' + roomname + '_gt.txt'
                    filename_soft_mIoU = 'checkpoints/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + \
                                         test_area + '/' + roomname + '/' + roomname + '_soft_' + mIoU_soft + '.txt'
                    # os.rename(filename_crd_mIoU, filename_soft_mIoU)
                    print('TXT visualization file saved in', filename_soft_mIoU)
                    print('TXT visualization file saved in', filename_crd_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % visu_format)
                exit()
    return visual_warning


def train(args, io):
    test_set = S3DIS(args.root, partition='test', num_points=args.num_points, test_area=args.test_area)
    test_loader = DataLoader(test_set, num_workers=4, batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    train_set = S3DIS(args.root, partition='train', num_points=args.num_points, test_area=args.test_area)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        net = DGCNNSegS3dis(args, pretrain=False).to(device)
    else:
        raise Exception("Not implemented")

    net = nn.DataParallel(net)
    if args.restore:
        point_model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus)
        model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', args.pretrained_path)
        point_model.load_state_dict(torch.load(model_path), strict=False)
        model = point_model.backbone
    else:
        model = net
    print("Model Loaded!!!")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)
    else:
        raise NotImplementedError

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        for data, seg in tqdm(train_loader, leave=False):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = ('Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f'
                  % (epoch, train_loss * 1.0 / count, train_acc, avg_per_class_acc, np.mean(train_ious)))
        io.cprint(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        torch.cuda.empty_cache()
        for data, seg in tqdm(test_loader, leave=False):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = ('Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f'
                  % (epoch, test_loss * 1.0 / count, test_acc, avg_per_class_acc, np.mean(test_ious)))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, args.test_area))
        torch.cuda.empty_cache()


def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1, 7):
        visual_file_index = 0
        test_area = str(test_area)
        if os.path.exists(f"{args.root}/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open(f"{args.root}/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
                    visual_file_index = visual_file_index + 1
        if (args.test_area == 'all') or (test_area == args.test_area):
            print('The visualization area is:', test_area)
            test_loader = DataLoader(S3DIS(
                args.root, partition='test', num_points=args.num_points, test_area=test_area),
                batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            device = torch.device("cuda" if args.cuda else "cpu")

            # Try to load models
            semseg_colors = test_loader.dataset.semseg_colors
            if args.model == 'dgcnn':
                model = DGCNNSegS3dis(args).to(device)
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area)))
            # model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_5.t7')))
            model = model.eval()
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            visual_warning = False
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                seg_crd = 0.48 * F.one_hot(seg, num_classes=13).to(seg_pred) + 0.52 * F.softmax(seg_pred / 0.5, dim=-1)
                seg_pred = 0.45*F.one_hot(seg, num_classes=13).to(seg_pred) + 0.55*F.softmax(seg_pred / 0.5, dim=-1)
                pred = seg_pred.max(dim=2)[1]
                crd_pred = seg_crd.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # visiualization
                visual_warning = visualization(args.root, args.visu, args.visu_format, args.test_area, data,
                                               seg, pred, crd_pred, visual_file_index, semseg_colors)
                visual_file_index = visual_file_index + data.shape[0]
            if visual_warning and args.visu != '':
                print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = ('Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f'
                      % (test_area, test_acc, avg_per_class_acc, np.mean(test_ious)))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = ('Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f'
                  % (all_acc, avg_per_class_acc, np.mean(all_ious)))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--root', type=str, default='/data/disk1/data/Stanford3dDataset', metavar='N',
                        help='path of dataset')
    parser.add_argument('--exp_name', type=str, default='dgcnn_SegS3dis', metavar='N',
                        help='Name of the experiment')  # dgcnn_SegS3dis
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--num_clus', type=int, default=64, metavar='N',
                        help='Num of clusters to use')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='restore the weights [default: True]')
    parser.add_argument('--test_area', type=str, default='5', metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--pretrained_path', type=str, default='checkpoints/dgcnn_segs3dis/models/best_model.pth',
                        metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_root', type=str,
                        default='checkpoints/dgcnn_SegS3dis/models', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='area_5',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
