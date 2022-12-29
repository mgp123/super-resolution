import torch
from torch import nn
import pytorch_lightning as pl

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm=32) -> None:
        super(Block, self).__init__()

        print(f"block: in_channels {in_channels}, out_channels {out_channels} ")

        self.model = nn.Sequential(
            nn.GroupNorm(group_norm,in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )
    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, group_norm=32):
        super(ResidualBlock, self).__init__()

        self.block1 = Block(
            in_channels=in_channels,
            out_channels=out_channels,
            group_norm=group_norm,
            )

        self.block2 = Block(
            in_channels=out_channels,
            out_channels=out_channels,
            group_norm=group_norm,
            )
        
        self.embedding = nn.Linear(embedding_size,out_channels)

        self.skip = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        
    def forward(self, x, v):
        y =  self.block1(x)
        v_emb = self.embedding(v)
        y += v_emb.view(y.shape[0], y.shape[1], 1, 1)
        y = self.block2(y)

        return y + self.skip(x)


class ResidualBlockDown(ResidualBlock):
    def __init__(self, in_channels, out_channels, embedding_size, group_norm=32):
        super(ResidualBlockDown, self).__init__(in_channels, out_channels, embedding_size, group_norm)

        self.down_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x, v):
        return self.down_conv(super().forward(x,v))

class ResidualBlockUp(ResidualBlock):
    def __init__(self, in_channels, out_channels, embedding_size, group_norm=32):
        super(ResidualBlockUp, self).__init__(in_channels, out_channels, embedding_size, group_norm)

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1)
        )

    def forward(self, x, v):
        return self.up_conv(super().forward(x,v))
