import torch
import numpy as np


def compute_feature_stats(dataset, indices):

    n = 0
    mean = None
    M2 = None

    for idx in indices:

        sample = dataset[idx]

        x = sample["features"]

        # remove invalid values
        x = x[torch.isfinite(x).all(dim=1)]

        x = x.numpy()

        if mean is None:
            n_features = x.shape[1]
            mean = np.zeros(n_features)
            M2 = np.zeros(n_features)


        for row in x:

            n += 1

            delta = row - mean
            mean += delta / n

            delta2 = row - mean
            M2 += delta * delta2


    std = np.sqrt(M2 / (n - 1))


    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32)
    )
