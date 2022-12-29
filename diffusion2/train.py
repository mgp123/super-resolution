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



data_loader_train, data_loader_test = get_data_loaders(batch_size=32, spatial_size=128)

# train
model = Diffusion(in_channels=6, out_channels=3, hidden_channels=64,scales=3,same_dimension_blocks=2)
logger = pl.loggers.TensorBoardLogger("runs", name="big_gan_diffusion")
trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, max_epochs=2)

trainer.fit(model, data_loader_train, data_loader_test)