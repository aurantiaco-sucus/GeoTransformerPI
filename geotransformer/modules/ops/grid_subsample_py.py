import math
from typing import Tuple

import torch


def grid_subsampling(points: torch.Tensor, lengths: torch.Tensor, voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python grid subsampling equivalent to the C++ extension.

    Args:
        points (Tensor): stacked points (N, 3)
        lengths (Tensor): number of points in each batch (B,)
        voxel_size (float): voxel size

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in each batch (B,)
    """
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)

    # Normalize dtypes/devices
    device = points.device
    dtype = points.dtype

    N = points.shape[0]

    # Ensure lengths is on CPU and a Python list of ints for iteration
    lengths_list = [int(x) for x in lengths.detach().cpu().tolist()]

    s_points_list = []
    s_lengths = []

    start = 0
    # We'll operate in CPU numpy for simple loops (works well for moderate point counts)
    points_cpu = points.detach().cpu().numpy()

    for l in lengths_list:
        if l == 0:
            s_lengths.append(0)
            continue

        cur = points_cpu[start:start + l]
        start += l

        # Compute bounding origin aligned to voxel grid
        min_corner = cur.min(axis=0)
        max_corner = cur.max(axis=0)
        origin = (math.floor(min_corner[0] / voxel_size) * voxel_size,
                  math.floor(min_corner[1] / voxel_size) * voxel_size,
                  math.floor(min_corner[2] / voxel_size) * voxel_size)

        sample_nx = int(math.floor((max_corner[0] - origin[0]) / voxel_size) + 1)
        sample_ny = int(math.floor((max_corner[1] - origin[1]) / voxel_size) + 1)

        # group points into voxels using a dict keyed by a linearized index
        voxels = {}
        for p in cur:
            ix = int(math.floor((p[0] - origin[0]) / voxel_size))
            iy = int(math.floor((p[1] - origin[1]) / voxel_size))
            iz = int(math.floor((p[2] - origin[2]) / voxel_size))
            key = ix + sample_nx * iy + sample_nx * sample_ny * iz

            if key not in voxels:
                voxels[key] = [p.copy(), 1]
            else:
                voxels[key][0] += p
                voxels[key][1] += 1

        # compute averaged point per voxel
        cur_s_points = [v[0] / v[1] for v in voxels.values()]
        s_points_list.extend(cur_s_points)
        s_lengths.append(len(cur_s_points))

    if len(s_points_list) == 0:
        s_points = torch.zeros((0, 3), dtype=dtype, device=device)
    else:
        import numpy as _np

        s_points = torch.as_tensor(_np.vstack(s_points_list), dtype=dtype, device=device)

    s_lengths = torch.as_tensor(s_lengths, dtype=torch.long, device=device)

    return s_points, s_lengths
