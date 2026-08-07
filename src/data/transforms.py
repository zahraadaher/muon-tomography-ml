import torch


class NormalizeFeatures:

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std.clone()

        self.std[self.std < 1e-8] = 1.0


    def __call__(self, x):

        return (x - self.mean) / self.std


class LogTarget:
    def __call__(self, target):
        return torch.log(target)
