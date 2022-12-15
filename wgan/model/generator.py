import torch
import torch.nn as nn

from model.dense_block import DenseBlock
from model.compressor import Compressor
from model.dimensional_layers import getConv

class Generator(nn.Module):
    def __init__(self, dimensions, in_channels, n_dense_blocks, layers_per_dense_block, out_channels_per_layer_dense=12):
        super(Generator, self).__init__()
        dense_blocks = []
        compressors = []
        in_channels_block = in_channels
        in_channels_compressor = 0
        out_channels_compressor = 2*out_channels_per_layer_dense

        for i in range(n_dense_blocks):

            block = DenseBlock(
                dimensions,
                in_channels=in_channels_block, 
                out_channels_per_layer=out_channels_per_layer_dense,
                n_convolutions=layers_per_dense_block
                )

            dense_blocks.append(block)

            in_channels_compressor += out_channels_per_layer_dense*layers_per_dense_block + in_channels_block
            if i < n_dense_blocks-1: 
                compresor = Compressor(
                    dimensions,
                    in_channels=in_channels_compressor, 
                    out_channels=out_channels_compressor)
                compressors.append(compresor)  

                in_channels_block = out_channels_compressor


        self.reconstruction = getConv(
                    dimensions,
                    in_channels=in_channels_compressor,
                    out_channels=in_channels,
                    kernel_size=1,
                )

        self.dense_blocks = nn.ModuleList(dense_blocks)
        self.compressors = nn.ModuleList([nn.Identity()] + compressors)
        # self.dense_blocks = self.dense_blocks.to("cuda:0")
        # self.compressors = self.compressors.to("cuda:0")

        # self.reconstruction = self.reconstruction.to("cuda:0")

    def forward(self, x):
        y = x

        for i in range(len(self.dense_blocks)):
            y2 = self.dense_blocks[i](self.compressors[i](y))
            if i != 0:
                y = torch.cat([y, y2], dim=1)
            else:
                y = y2

        return self.reconstruction(y)
