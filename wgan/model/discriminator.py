import torch
import torch.nn as nn

from model.dimensional_layers import getConv


def ceil_div(a,b):
    res = a//b
    if (res * b) < a:
        res += 1
    return res

class DiscriminatorBlock(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, spatial_size, stride=1, layer_normalization=True):
        super(DiscriminatorBlock, self).__init__()

        ln = nn.Identity()
        if layer_normalization:
            ln_spatial = ceil_div(spatial_size,stride)
            ln = nn.LayerNorm([out_channels] + [ln_spatial]*dimensions)


        self.model = nn.Sequential(
            getConv(
                dimensions,
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3,
                padding=1,
                stride=stride),
            ln,
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, dimensions, in_channels, spatial_size):
        super(Discriminator, self).__init__()

        cutoff_size = 3
        current_spatial_size = spatial_size
        current_in_channels = in_channels
        layers = []
        layers.append(
            DiscriminatorBlock(
                dimensions,
                in_channels=current_in_channels, 
                out_channels=64, 
                spatial_size=current_spatial_size,
                layer_normalization=False)
        )

        current_in_channels = 64

        layers.append(
            DiscriminatorBlock(
                dimensions,
                in_channels=current_in_channels, 
                out_channels=64,
                spatial_size=current_spatial_size, 
                stride=2)
        )

        current_spatial_size = ceil_div(current_spatial_size, 2)

        for i in range(3):

            layers.append(
                DiscriminatorBlock(
                    dimensions,
                    in_channels=current_in_channels, 
                    out_channels=current_in_channels, 
                    spatial_size=current_spatial_size)
            )

            layers.append(
                DiscriminatorBlock(
                    dimensions,
                    in_channels=current_in_channels, 
                    out_channels=current_in_channels*2,
                    spatial_size=current_spatial_size, 
                    stride=2)
            )

            current_spatial_size = ceil_div(current_spatial_size, 2)
            current_in_channels = current_in_channels*2

        layers.append(torch.nn.Flatten())
        layers.append(nn.Linear(current_in_channels*(current_spatial_size**dimensions), 1024))
        layers.append(torch.nn.LeakyReLU())
        layers.append(nn.Linear(1024, 1))
        
        # TODO should it end in Sigmoid if this is a WGAN?
        # layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)