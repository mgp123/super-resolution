import torch
import torch.nn as nn

from model.dimensional_layers import getConv

class Compressor(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels):
        super(Compressor, self).__init__()
        self.model = getConv(
            dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            )

    def forward(self, x):
        return self.model(x)
