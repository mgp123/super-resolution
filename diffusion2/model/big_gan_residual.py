import torch
from torch import nn
import pytorch_lightning as pl

from model.blurr_pool import MaxBlurPool



class BigGanResidual(nn.Module):
    """
    BigGan Residual Down is a residual block. It halves spatial dimensions and double channels
    """
    def __init__(self, in_channels, out_channels, embedding_size):
        # print(f"Created BigGanResidualDown with channels {in_channels}")
        super(BigGanResidual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.spatial_transform = nn.Identity()

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )

        self.mid_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
        )

        if embedding_size is not None:
            self.embedding1 = nn.Linear(embedding_size,in_channels)
            self.embedding2 = nn.Linear(embedding_size,in_channels)

        self.skip = nn.Identity()

        

    def forward(self, x, v=None):
        y = x
        if v is not None:
            v_emb1 = self.embedding1(v)
            y =  y + v_emb1.view(x.shape[0], x.shape[1], 1, 1)

        y = self.block1(y)
        
        y = self.spatial_transform(y)
        y = self.mid_conv(y)

        if v is not None:
            v_emb2 = self.embedding2(v)
            y =  y + v_emb2.view(y.shape[0], y.shape[1], 1, 1)
            
        y = self.block2(y)

        skip = self.skip(x)
        return y + (2**(-0.5))*skip



class BigGanResidualDown(BigGanResidual):
    def __init__(self, in_channels, out_channels, embedding_size):
        super(BigGanResidualDown,self).__init__(in_channels, out_channels, embedding_size)
        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1),
            MaxBlurPool()
        )
        self.spatial_transform = MaxBlurPool()

       
class BigGanResidualUp(BigGanResidual):
    def __init__(self, in_channels, out_channels, embedding_size):
        super().__init__(in_channels, out_channels, embedding_size)

        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1),
            nn.Upsample(scale_factor=2)
        )
        self.spatial_transform = nn.Upsample(scale_factor=2)


class BigGanResidualSame(BigGanResidual):
    def __init__(self, in_channels, out_channels, embedding_size):
        super().__init__(in_channels, out_channels, embedding_size)

        self.skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )


# m = BigGanResidualSame(in_channels=32)

# x = torch.randn((4,32,128,128))
# y = m(x)
# print(y.shape)
