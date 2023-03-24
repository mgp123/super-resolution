import math
import torch
from torch import nn
import lightning.pytorch as pl
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils import data
from typing import Tuple

from data_loader import BrainDataset
from gan import WGAN



training_parameters = {
    "batch_size": 8,
    "low_res_size": (80,128,128),
    "kernel_size": (16,16,16), 
    "coefficient_reconstruction_loss" : 0.99,
    "coefficient_perceptual_loss": 0.01,
    "discriminator_iterations_per_batch": 2,
    "coefficient_lipschitz_loss":  1e-2,
    "lr_g": 3e-4,
    "lr_d": 3e-4,
    "betas" : (0.5,0.999)
    }

generator_config = {
    "dimensions" :  3,
    "in_channels" :  1,
    "out_channels" :  1, 
    "n_dense_blocks": 3,
    "layers_per_dense_block": 2, 
    "out_channels_per_layer_dense": 12
}


discriminator_config = {
    "dimensions": generator_config["dimensions"],
    "in_channels": generator_config["out_channels"], 
    "spatial_size": training_parameters["kernel_size"]
}


config = {
    "training_parameters" : training_parameters,
    "generator_config": generator_config,
    "discriminator_config": discriminator_config
}

data_loader_train = data.DataLoader(
    BrainDataset("local", training_parameters["kernel_size"], training_parameters["low_res_size"]),
    batch_size=training_parameters["batch_size"],
    num_workers=12,
    shuffle=True,
    pin_memory=True
)


# train
model = WGAN(config)

logger = pl.loggers.TensorBoardLogger("runs")
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1, 
    logger=logger,
    precision=16,
    max_epochs=30
    )

trainer.fit(
    model, 
    data_loader_train, 
    )