"""Space-filling-curve keys that serialize point clouds for window attention.

Keys are computed from integer voxel coordinates with ``bits`` bits per axis.
At a finer resolution both curves refine the order of coarser cells, so a
fine key sorts points exactly like the coarse key and breaks its ties
spatially.
"""

import torch

CURVES = ("z", "hilbert")


def morton_key(grid, bits):
    """Z-order (Morton) key of ... x 3 integer coordinates in [0, 2**bits): octree node order."""
    key = torch.zeros_like(grid[..., 0])
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            key = (key << 1) | ((grid[..., axis] >> bit) & 1)
    return key


def hilbert_key(grid, bits):
    """3-D Hilbert key of ... x 3 integer coordinates in [0, 2**bits).

    Skilling's transpose algorithm ("Programming the Hilbert curve", AIP
    Conference Proceedings 707, 2004), vectorized over points.
    """
    x = [grid[..., axis].clone() for axis in range(3)]
    top = 1 << (bits - 1)
    q = top
    while q > 1:  # Inverse undo.
        p = q - 1
        for axis in range(3):
            high = (x[axis] & q) != 0
            swap = torch.where(high, torch.zeros_like(x[0]), (x[0] ^ x[axis]) & p)
            x[0] = torch.where(high, x[0] ^ p, x[0] ^ swap)
            if axis:
                x[axis] = x[axis] ^ swap
        q >>= 1
    for axis in range(1, 3):  # Gray encode.
        x[axis] = x[axis] ^ x[axis - 1]
    t = torch.zeros_like(x[0])
    q = top
    while q > 1:
        t = torch.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x = [value ^ t for value in x]
    key = torch.zeros_like(x[0])
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            key = (key << 1) | ((x[axis] >> bit) & 1)
    return key


def curve_key(grid, bits, curve="z"):
    if curve == "z":
        return morton_key(grid, bits)
    if curve == "hilbert":
        return hilbert_key(grid, bits)
    raise ValueError(f"curve must be one of {CURVES}")
