import numpy as np
import torch


def radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius):
    """Pure-Python implementation of radius_neighbors used as a fallback when
    the native extension is not available.

    Args:
        q_points (Tensor): (N, 3) stacked query points
        s_points (Tensor): (M, 3) stacked support points
        q_lengths (LongTensor): (B,) lengths of each batch element in q_points
        s_lengths (LongTensor): (B,) lengths of each batch element in s_points
        radius (float): search radius

    Returns:
        Tensor: (N, max_neighbors) long tensor of neighbor indices into s_points.
            Rows are padded with M (total number of support points) when fewer
            than max_neighbors neighbors are found.
    """
    # Move to CPU numpy arrays for processing
    device = q_points.device
    q_pts = q_points.detach().cpu().numpy().astype(np.float32)
    s_pts = s_points.detach().cpu().numpy().astype(np.float32)

    q_lens = q_lengths.detach().cpu().numpy().astype(np.int64)
    s_lens = s_lengths.detach().cpu().numpy().astype(np.int64)

    total_q = q_pts.shape[0]
    total_s = s_pts.shape[0]
    r = float(radius)
    r2 = r * r

    # Prepare per-query neighbor lists
    neighbors = [[] for _ in range(total_q)]

    q_start = 0
    s_start = 0
    for b in range(len(q_lens)):
        q_end = q_start + int(q_lens[b])
        s_end = s_start + int(s_lens[b])

        if s_end <= s_start:
            # No support points in this batch element
            for i in range(q_start, q_end):
                neighbors[i] = []
        else:
            q_batch = q_pts[q_start:q_end]
            s_batch = s_pts[s_start:s_end]

            # Try to use scipy cKDTree for performance; fall back to brute force
            try:
                from scipy.spatial import cKDTree as KDTree
                tree = KDTree(s_batch)
                idxs = tree.query_ball_point(q_batch, r)
                for i_local, idx_list in enumerate(idxs):
                    global_i = q_start + i_local
                    if len(idx_list) == 0:
                        neighbors[global_i] = []
                    else:
                        # compute distances and sort
                        pts_neighbors = s_batch[idx_list]
                        diffs = pts_neighbors - q_batch[i_local:i_local+1]
                        dists = np.sum(diffs * diffs, axis=1)
                        order = np.argsort(dists)
                        neighbors[global_i] = [s_start + int(idx_list[j]) for j in order]
            except Exception:
                # Brute-force fallback: compute full distance matrix for this batch
                # (q_batch_size, s_batch_size)
                diffs = q_batch[:, None, :] - s_batch[None, :, :]
                dists = np.sum(diffs * diffs, axis=2)
                for i_local in range(q_end - q_start):
                    ds = dists[i_local]
                    inds = np.nonzero(ds <= r2)[0]
                    if inds.size == 0:
                        neighbors[q_start + i_local] = []
                    else:
                        order = np.argsort(ds[inds])
                        sorted_local = inds[order]
                        neighbors[q_start + i_local] = [s_start + int(j) for j in sorted_local]

        q_start = q_end
        s_start = s_end

    # Determine maximum neighbor count
    max_count = 0
    for row in neighbors:
        if len(row) > max_count:
            max_count = len(row)

    if max_count == 0:
        # Return empty (N, 0) tensor with long dtype
        return torch.zeros((total_q, 0), dtype=torch.long, device=device)

    out = np.full((total_q, max_count), total_s, dtype=np.int64)
    for i, row in enumerate(neighbors):
        if len(row) > 0:
            out[i, : len(row)] = row

    return torch.from_numpy(out).to(device)
