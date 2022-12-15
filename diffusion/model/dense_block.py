import torch
import torch.nn as nn


# adapted from another project of mine
class DenseBlock(nn.Module):
    def __init__(self, in_channels, n_convolutions, out_channels_per_layer = 12, kernel_size = 3, padding = 1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_convolutions):
            in_channels_layer = in_channels + i*out_channels_per_layer
            bottleneck_layer = nn.Identity()
            # This adds an extra convolution that reduces channels if in_channels is too high
            if in_channels_layer > out_channels_per_layer*4:
                bottleneck_layer = nn.Conv3d(
                    in_channels = in_channels_layer,
                    out_channels= out_channels_per_layer*4,
                    kernel_size=1
                )
    
                in_channels_layer = out_channels_per_layer*4


            layer = nn.Sequential(
                bottleneck_layer,
                nn.BatchNorm2d(in_channels_layer),
                nn.ELU(),
                nn.Conv2d(
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
