import importlib


try:
    ext_module = importlib.import_module('geotransformer.ext')
    _grid_subsampling = ext_module.grid_subsampling
except Exception:
    # Fallback to pure Python implementation when extension is not available
    from . import grid_subsample_py as _py_impl
    _grid_subsampling = _py_impl.grid_subsampling


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function tries the compiled extension first, and falls back to a pure-Python
    implementation implemented in `grid_subsample_py.py`.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_lengths = _grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths
