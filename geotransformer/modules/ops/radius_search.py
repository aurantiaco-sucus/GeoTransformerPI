import importlib


try:
    ext_module = importlib.import_module('geotransformer.ext')
    _radius_neighbors = ext_module.radius_neighbors
except Exception:
    # Fallback to pure Python implementation when extension is not available
    from . import radius_search_py as _py_impl
    _radius_neighbors = _py_impl.radius_neighbors


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function tries a compiled extension first and falls back to a pure-Python
    implementation implemented in `radius_search_py.py`.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    neighbor_indices = _radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices
