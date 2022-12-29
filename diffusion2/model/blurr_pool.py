import torch
from torch import nn
import pytorch_lightning as pl

## code (somewhat) adapted from antialiased-cnns

class BlurPool(pl.LightningModule):
    def __init__(self):
        super(BlurPool, self).__init__()
        blurr_kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]])
        blurr_kernel = blurr_kernel / torch.sum(blurr_kernel)
        self.register_buffer('blurr_kernel', blurr_kernel)

    def forward(self, x):
        channels = x.shape[1]
        y = nn.functional.conv2d(x, self.blurr_kernel.repeat((channels,1,1,1)), stride=2, groups=channels)
        return y

class MaxBlurPool(pl.LightningModule):
    def __init__(self, max_kernel_size=2):
        super(MaxBlurPool, self).__init__()
        self.blurry_pool = BlurPool()
        self.max_pool = nn.MaxPool2d(kernel_size=max_kernel_size,stride=1, padding=max_kernel_size//2 )
    
    def forward(self, x):
        y = self.max_pool(x)
        y = self.blurry_pool(y)
        return y

