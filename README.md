# CluRender: Unsupervised Point Cloud Representation Learning by Clustering and Neural Rendering

Official PyTorch implementation of
[**Unsupervised Point Cloud Representation Learning by Clustering and Neural Rendering**](https://doi.org/10.1007/s11263-024-02027-5),
*International Journal of Computer Vision* 132, 3251–3269 (2024).

Guofeng Mei, Cristiano Saltori, Elisa Ricci, Nicu Sebe, Qiang Wu, Jian Zhang, Fabio Poiesi

## Introduction

CluRender learns transferable point-level features without data augmentation.
Two self-supervised objectives train the same encoder:

- **Soft clustering.** A head predicts cluster assignments for every point.
  Balanced pseudo-labels, obtained with Sinkhorn optimal transport, divide each
  point cloud into approximately equal partitions, and a cross-entropy loss fits
  the predictions to them.
- **Neural rendering.** A color decoder predicts an RGB value for every point.
  The colored points are splatted into calibrated views, and an optimal
  transport loss measures the consistency between rendered and real images.

The pretrained encoder transfers to classification, part segmentation, semantic
segmentation, object detection and few-shot learning.

<details>
<summary>Abstract</summary>

Data augmentation has contributed to the rapid advancement of unsupervised
learning on 3D point clouds. However, we argue that data augmentation is not
ideal, as it requires a careful application-dependent selection of the types of
augmentations to be performed, thus potentially biasing the information learned
by the network during self-training. Moreover, several unsupervised methods only
focus on unimodal information, thus potentially introducing challenges in the
case of sparse point clouds. To address these issues, we propose an
augmentation-free unsupervised approach for point clouds, named CluRender, to
learn transferable point-level features by leveraging unimodal information for
soft clustering and cross-modal information for neural rendering. Soft
clustering enables self-training through a pseudo-label prediction task, where
the affiliation of points to their clusters is used as a proxy under the
constraint that these pseudo-labels divide the point cloud into approximate
equal partitions. This allows us to formulate a clustering loss to minimize the
standard cross-entropy between pseudo and predicted labels. Neural rendering
generates photorealistic renderings from various viewpoints to transfer
photometric cues from 2D images to the features. The consistency between
rendered and real images is then measured to form a fitting loss, combined with
the cross-entropy loss to self-train networks. Experiments on downstream
applications, including 3D object detection, semantic segmentation,
classification, part segmentation, and few-shot learning, demonstrate the
effectiveness of our framework in outperforming state-of-the-art techniques.

</details>

## Installation

The code requires Python 3.10 or newer and PyTorch 2.1 or newer with CUDA.

```bash
git clone https://github.com/gfmei/clurender.git
cd clurender
pip install -r requirements-pretrain.txt   # NumPy, Pillow, SciPy, GeomLoss, KeOps
pip install h5py scikit-learn              # downstream evaluation
```

Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
for your PyTorch and CUDA versions. It is used for point rendering during
pretraining and for rendering the pretraining images.
[KeOps](https://www.kernel-operations.io/keops/) is only needed for the optional
Sinkhorn image loss (`--image-loss sinkhorn`); it needs a C++ compiler and the
CUDA toolkit, and compiles its kernels on first use.

## Datasets

We use the following directory layout:

```text
data/
├── ShapeNetCore/                    # ShapeNetCore v2 meshes, for pretraining
├── shapenet_paired/                 # prepared pretraining data (see below)
├── modelnet40_ply_hdf5_2048/
├── ScanObjectNN/
│   ├── main_split/
│   └── main_split_nobg/
├── ModelNetFewshot/
│   ├── 5way10shot/  5way20shot/  10way10shot/  10way20shot/
└── shapenetcore_partanno_segmentation_benchmark_v0_normal/
```

**ShapeNetCore (pretraining).** Request access to
[ShapeNetCore on Hugging Face](https://huggingface.co/datasets/ShapeNet/ShapeNetCore),
then download and extract the 55 category archives:

```bash
hf download ShapeNet/ShapeNetCore --repo-type dataset --include "*.zip" --local-dir data/ShapeNetCore/zips
for z in data/ShapeNetCore/zips/*.zip; do unzip -q -n "$z" -x '*.binvox' -d data/ShapeNetCore; done
```

**Downstream datasets.**

| Dataset | Source |
| --- | --- |
| ModelNet40 | The `modelnet40_ply_hdf5_2048` release of PointNet and DGCNN |
| ScanObjectNN | [Official website](https://hkust-vgd.github.io/scanobjectnn/) (`main_split`, `main_split_nobg`) |
| ModelNet40 few-shot | Splits of [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) (`ModelNetFewshot`) |
| ShapeNetPart | [`shapenetcore_partanno_segmentation_benchmark_v0_normal`](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) |

### Preparing the pretraining data

`prepare_shapenet.py` pairs every ShapeNetCore mesh with calibrated RGB views.
Each mesh is centered and scaled to unit radius and rendered with PyTorch3D
from cameras on an orbit around the object (distance 2.7, elevation 15°,
60° field of view).

```bash
python prepare_shapenet.py --root data/ShapeNetCore --output data/shapenet_paired --views 8
```

Each object is stored as `data/shapenet_paired/<category>/<model>.npz`:

| Array | Shape | Content |
| --- | --- | --- |
| `points` | `(N, 3)` | Surface points (default N = 8192) |
| `normals` | `(N, 3)` | Unit surface normals |
| `colors` | `(N, 3)` | RGB color of each point |
| `images` | `(V, H, W, 3)` | RGB views, uint8 |
| `depths` | `(V, H, W)` | Camera-space depth; 0 for background |
| `visibility` | `(V, N)` | Whether each view sees each point |
| `tsfms` | `(V, 4, 4)` | OpenCV world-to-camera transforms |
| `K` | `(V, 3, 3)` | Pinhole intrinsics in pixels |

`--split` restricts preparation to listed `category/model` identifiers, and
`--shard INDEX COUNT` distributes the work over several processes.

Other paired data can be used as well. Pretraining needs only `points`,
`images`, `tsfms` and `K` in each archive. Cameras follow the OpenCV convention
(X right, Y down, Z forward), with the center of the top-left pixel at `(0, 0)`,
and points and cameras share one world frame.

`datasets.multiview.project_points` returns the pixel coordinates of points in
every view, which, together with `visibility`, lifts per-pixel image features
onto the points.

## Pretraining

```bash
torchrun --standalone --nproc_per_node=4 main_pretrain.py \
  --root data/shapenet_paired --output checkpoints/clurender --model dgcnn \
  --batch-size 32 --num-points 1024 --num-views 8 --epochs 250 --device cuda
```

`--batch-size` is the total over all GPUs; on a single GPU, replace
`torchrun --standalone --nproc_per_node=4` with `python`. Every `.npz` file
under `--root` is used, unless `--split` lists archive paths relative to the
root. Training resumes with
`--resume checkpoints/clurender/last.pth`, where `--epochs` is the total number
of epochs. The main settings are:

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `dgcnn` | Encoder: `dgcnn` or `pointnet` |
| `--num-clusters` | 64 | Clusters per point cloud |
| `--num-views`, `--image-size` | 8, 256 | Views per object and their resolution |
| `--epsilon` | 0.001 | Entropy of the clustering Sinkhorn |
| `--orthogonal-weight` | 0.01 | Weight of the prototype regularizer |
| `--render-weight` | 1.0 | Weight of the rendering loss |
| `--image-loss`, `--num-projections` | `sliced`, 128 | Sliced Wasserstein with 128 projections, or `sinkhorn` (entropic OT) |
| `--lr`, `--lr-step`, `--lr-gamma` | 0.001, 20, 0.7 | AdamW with step decay |

`python main_pretrain.py --help` lists all options. The output directory
contains full checkpoints (`last.pth`, `best.pth`, `epoch_NNNN.pth`), the
encoder weights `backbone.pth`, the settings and per-epoch metrics.

## Downstream Tasks

All downstream scripts initialize the encoder from `backbone.pth`; omit
`--pretrained` to train the same network from scratch. Classification and
segmentation follow the evaluation protocol of
[Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

### Linear SVM

A linear SVM is trained on frozen global features of the pretrained encoder.

```bash
python main_svm.py --dataset modelnet40 --root data/modelnet40_ply_hdf5_2048 \
  --pretrained checkpoints/clurender/backbone.pth
```

### Classification

```bash
# ModelNet40
python main_finetune_cls.py --dataset modelnet40 --root data/modelnet40_ply_hdf5_2048 \
  --pretrained checkpoints/clurender/backbone.pth --output runs/modelnet40

# ScanObjectNN: --variant objbg, objonly or hardest (PB-T50-RS)
python main_finetune_cls.py --dataset scanobjectnn --variant hardest --root data/ScanObjectNN \
  --point-pool 2048 --pretrained checkpoints/clurender/backbone.pth --output runs/scanobjectnn_hardest
```

The model is tested after every epoch, and the checkpoint with the highest
accuracy is finally evaluated with 10-vote augmentation. `results.json` reports
the last epoch, the best epoch and the voting accuracy.

### Few-shot Classification

```bash
python main_finetune_cls.py --dataset fewshot --root data/ModelNetFewshot \
  --way 5 --shot 10 --fold 0 --epochs 150 \
  --pretrained checkpoints/clurender/backbone.pth --output runs/fewshot/5way10shot/fold0
```

Run folds 0–9 for each of the 5-way/10-way, 10-shot/20-shot settings and report
the mean and standard deviation of the accuracy.

### Part Segmentation

```bash
python main_finetune_partseg.py --root data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
  --exclude-overlap --pretrained checkpoints/clurender/backbone.pth --output runs/partseg
```

Predictions are restricted to the parts of each object's category. The script
reports point accuracy, class mIoU and instance mIoU. `--exclude-overlap`
removes the objects that the train/val lists repeat or share with the test list
from training.

`main_partseg_probe.py` provides a frozen-encoder alternative, which trains
only a linear per-point classifier on the pretrained features.

### Using the Pretrained Encoder

```python
import torch
from models.dgcnn import DGCNN

encoder = DGCNN(emb_dims=1024, k=20, num_cls=-1)
encoder.load_state_dict(torch.load("checkpoints/clurender/backbone.pth", map_location="cpu"))
global_features, point_features = encoder(points)  # points: B x 3 x N
```

## Implementation Details

- **Clustering.** Pseudo-labels come from log-domain Sinkhorn iterations
  (entropy 0.001), which run until the point marginals are within 1% of balance,
  followed by a projection onto the exact marginals
  ([Altschuler et al., 2017](https://arxiv.org/abs/1705.09634)).
  `--sinkhorn-iterations 20 --sinkhorn-tolerance 0` instead runs a fixed budget
  of 20 iterations.
- **Color decoder.** A point-transformer U-Net with local vector attention
  (16 neighbors) and three resolution levels (N, N/4, N/16).
- **Rendering.** Points are splatted with PyTorch3D as Gaussian disks
  (4-pixel radius, σ = 2 pixels) and composited in depth order.
- **Image loss.** Each image is a set of samples, one per pixel, of its color
  and position. Rendered and target images are compared with the sliced
  Wasserstein distance over 128 random projections. Its cost and memory grow
  linearly with the number of pixels: no pixel-to-pixel cost matrix is formed,
  and the gradient is accumulated while sorting, so full 256 × 256 images are
  used. `--image-loss sinkhorn` instead uses entropy-regularized optimal
  transport ([GeomLoss](https://www.kernel-operations.io/geomloss/) with KeOps,
  blur 0.01).
- **Pretraining images.** `prepare_shapenet.py` renders ShapeNet with PyTorch3D;
  these images differ from the renderings used in the paper.

## Citation

If you find this work useful, please cite:

```bibtex
@article{mei2024clurender,
  title   = {Unsupervised Point Cloud Representation Learning by Clustering and Neural Rendering},
  author  = {Mei, Guofeng and Saltori, Cristiano and Ricci, Elisa and Sebe, Nicu and
             Wu, Qiang and Zhang, Jian and Poiesi, Fabio},
  journal = {International Journal of Computer Vision},
  volume  = {132},
  number  = {8},
  pages   = {3251--3269},
  year    = {2024},
  doi     = {10.1007/s11263-024-02027-5}
}
```

## Acknowledgements

This repository builds on [DGCNN](https://github.com/WangYueFt/dgcnn),
[Point-MAE](https://github.com/Pang-Yatian/Point-MAE),
[Point-BERT](https://github.com/lulutang0608/Point-BERT),
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) and
[GeomLoss](https://github.com/jeanfeydy/geomloss).
