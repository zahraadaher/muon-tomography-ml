import os
import numpy as np
import torch


def _filter_to_voi(features, metadata):
    """Keep only points inside the voxelized volume, matching prepare()."""
    xyz = features[:, :3]
    xmin, ymin, zmin = metadata["voxel_origin"]
    voxel_dims = metadata["voxel_shape"]
    voxel_size = metadata["voxel_size"]
    xmax = xmin + voxel_dims[0] * voxel_size
    ymax = ymin + voxel_dims[1] * voxel_size
    zmax = zmin + voxel_dims[2] * voxel_size

    inside = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )
    return features[inside]


def compute_feature_stats(reader, indices, metadata=None):
    """
    Stream samples from the HDF5Reader for the given indices and compute
    per-feature mean/std, vectorized (no full-dataset caching required).
    """
    n_features = None
    total_sum = None
    total_sumsq = None
    total_count = 0

    with reader:
        for name in indices:

            x = reader.get_sample(name)["features"]

            if metadata is not None:
                x = _filter_to_voi(x, metadata)

            finite_mask = np.isfinite(x).all(axis=1)
            x = x[finite_mask]

            if x.shape[0] == 0:
                continue

            if n_features is None:
                n_features = x.shape[1]
                total_sum = np.zeros(n_features, dtype=np.float64)
                total_sumsq = np.zeros(n_features, dtype=np.float64)

            x64 = x.astype(np.float64)
            total_sum += x64.sum(axis=0)
            total_sumsq += (x64 ** 2).sum(axis=0)
            total_count += x64.shape[0]

    if total_count < 2:
        raise RuntimeError("Not enough valid rows to compute feature statistics.")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var * total_count / (total_count - 1))

    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def get_or_compute_feature_stats(reader, indices, split_path, metadata=None):
    """
    Only computes feature stats if no cached stats file exists yet for
    this split. The stats file is derived from split_path, e.g.
    'default_split.json' -> 'default_split_feat_stats.npz'.
    """
    stats_path = os.path.splitext(split_path)[0] + "_feat_stats.npz"

    if os.path.exists(stats_path):
        print(f"Feature stats already exist, loading from {stats_path}")
        stats = np.load(stats_path)
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        return mean, std

    print(f"No cached feature stats found at {stats_path}, computing now...")
    mean, std = compute_feature_stats(reader, indices, metadata=metadata)

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    np.savez(stats_path, mean=mean.numpy(), std=std.numpy())
    print(f"Saved feature stats to {stats_path}")

    return mean, std