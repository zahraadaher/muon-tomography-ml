"""
Defines the voxel center positions in spatial coodinates, given the grid dimensions, 
voxel side length, and the spatial position of the corner of the grid corresponding to 
the voxel with index (0, 0, 0).
"""

import torch


def get_voxel_centers(
    voxel_shape,
    voxel_size,
    origin
):
    nx, ny, nz = voxel_shape
    ox, oy, oz = origin

    x = torch.arange(nx)
    y = torch.arange(ny)
    z = torch.arange(nz)

    xv, yv, zv = torch.meshgrid(
        x, y, z,
        indexing="ij"
    )

    centers = torch.stack(
        [xv, yv, zv],
        dim=-1
    ).float()

    centers = centers * voxel_size + voxel_size/2

    centers[..., 0] += ox
    centers[..., 1] += oy
    centers[..., 2] += oz

    return centers.reshape(-1, 3)
