import torch
import torch.nn as nn

from model.dimensional_layers import getBatchNorm, getConv

class DenseBlock(nn.Module):
    def __init__(self, dimensions, in_channels, n_convolutions, out_channels_per_layer = 12, kernel_size = 3, padding = 1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_convolutions):
            in_channels_layer = in_channels + i*out_channels_per_layer
            bottleneck_layer = nn.Identity()
            # This adds an extra convolution that reduces channels if in_channels is too high
            # TODO check if this is correct
            if in_channels_layer > out_channels_per_layer*4:
                bottleneck_layer = getConv(
                    dimensions,
                    in_channels = in_channels_layer,
                    out_channels= out_channels_per_layer*4,
                    kernel_size=1
                )
    
                in_channels_layer = out_channels_per_layer*4


            layer = nn.Sequential(
                bottleneck_layer,
                getBatchNorm(dimensions,num_features= in_channels_layer),
                nn.ELU(),
                getConv(
                    dimensions,
                    in_channels=in_channels_layer,
                    out_channels=out_channels_per_layer,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        y = x

        for layer in self.layers:
            y2 = layer(y)
            y = torch.cat([y, y2], dim=1)

        return y
