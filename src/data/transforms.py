import torch


class NormalizeFeatures:
    r"""
    Transfrom for standardizing feature inputs.
    Provided as an argument to :class:`MuonTomographyDataset.Dataset` class, if enabled. 

    Args:
    -----
    - mean: tensor of the feature mean values
    - std: tensor of the feature std values
    """
    def __init__(self, mean: torch.tensor, std: torch.tensor):

        self.mean = mean
        self.std = std.clone()

        self.std[self.std < 1e-8] = 1.0


    def __call__(self, x):

        return (x - self.mean) / self.std


class LogTarget:
    r'''
    Transforms ground truth target :math:`X_0` to :math:`log(X_0)`
    '''
    def __call__(self, target):
        return torch.log(target)


class LogInverseTarget:
    r'''
    Transforms ground truth target :math:`X_0` to :math:`1/log(X_0)`
        '''
    def __call__(self, target):
        return torch.log(1/target)
