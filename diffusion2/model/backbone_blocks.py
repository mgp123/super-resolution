import torch
from torch import nn

from model.big_gan_residual import BigGanResidualDown, BigGanResidualSame, BigGanResidualUp

class BackboneBlocks:
    def __init__(self, backbone_type) -> None:
        if backbone_type == "big_gan":
            self.up = BigGanResidualUp
            self.down = BigGanResidualDown
            self.same = BigGanResidualSame
        else:
            raise ValueError(f"BackboneBlocks does not recognize '{backbone_type}'")
