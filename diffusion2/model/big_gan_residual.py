import torch
from torch import nn
import pytorch_lightning as pl

from model.blurr_pool import MaxBlurPool



# kind of implmentation of big gan residual, although its not that similar
class BigGanResidualDown(pl.LightningModule):
    """
    BigGan Residual Down is a residual block. It halves spatial dimensions and double channels
    """
    def __init__(self, in_channels):
        # print(f"Created BigGanResidualDown with channels {in_channels}")

        out_channels=in_channels*2
        super(BigGanResidualDown, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            MaxBlurPool(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.blur_pool =  MaxBlurPool()
        self.conv = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            stride=2
        )
    def forward(self, x):
        y = self.model(x)
        skip = torch.cat([self.blur_pool(x), self.conv(x)], dim=1)
        return y + (2**(-0.5))*skip



class BigGanResidualUp(pl.LightningModule):
    """
    BigGan Residual Up is a residual block. It doubles spatial dimensions and halves channels
    """
    def __init__(self, in_channels):
        # print(f"Created BigGanResidualUp with channels {in_channels}")
        out_channels=in_channels//2
        super(BigGanResidualUp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.upsample = torch.nn.Upsample(scale_factor=2)

        self.conv = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    def forward(self, x):
        y = self.model(x)
        # TODO is this a valid way to drop some channels?
        skip = self.upsample(x[:,:x.shape[1]//2])
        return y + (2**(-0.5))*skip



class BigGanResidualSame(pl.LightningModule):
    """
    BigGan Residual Same is a residual block.
    """
    def __init__(self, in_channels):
        super(BigGanResidualSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )


    def forward(self, x):
        return self.model(x) + (2**(-0.5))*x




# m = BigGanResidualSame(in_channels=32)

# x = torch.randn((4,32,128,128))
# y = m(x)
# print(y.shape)
