import math
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils import data
from data_loaders import get_data_loaders
from model.model import Diffusion
from typing import Tuple



data_loader_train, data_loader_test = get_data_loaders(batch_size=32, spatial_size=64)

# train
model = Diffusion(
    in_channels=3, 
    out_channels=3, 
    hidden_channels=128,
    scales=(1,2,2,2,1),
    attention=True,
    context_encoding=None
    )
logger = pl.loggers.TensorBoardLogger("runs", name="diffusion_generator")
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1, 
    logger=logger, 
    max_epochs=6, 
    )

trainer.fit(
    model, 
    data_loader_train, 
    data_loader_test,
    ckpt_path="runs/diffusion_generator/version_4/checkpoints/epoch=3-step=13122.ckpt" 
    )