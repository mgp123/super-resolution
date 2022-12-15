import torch.nn as nn

def getBatchNorm(d, **kwargs):
    if d == 1:
        return nn.BatchNorm1d(**kwargs)
    elif d == 2:
        return nn.BatchNorm2d(**kwargs)
    elif d == 3:
        return nn.BatchNorm3d(**kwargs)
    else:
        raise NotImplementedError(f"Batchnorm for dimension {d} must be implemented by hand")

def getConv(d, **kwargs):
    if d == 1:
        return nn.Conv1d(**kwargs)
    elif d == 2:
        return nn.Conv2d(**kwargs)
    elif d == 3:
        return nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError(f"Conv for dimension {d} must be implemented by hand")


def getConvTransposed(d, **kwargs):
    if d == 1:
        return nn.ConvTranspose1d(**kwargs)
    elif d == 2:
        return nn.ConvTranspose2d(**kwargs)
    elif d == 3:
        return nn.ConvTranspose3d(**kwargs)
    else:
        raise NotImplementedError(f"ConvTranspose for dimension {d} must be implemented by hand")
