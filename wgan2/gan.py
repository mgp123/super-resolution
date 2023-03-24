import os
from typing import Any, Tuple
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from tqdm import tqdm

from model.discriminator import Discriminator
from model.generator import Generator
from nii_handler import Nii_Handler


class WGAN(pl.LightningModule):
    """
    Module containing both the discriminator and the generator and doing the training
    """

    def __init__(
        self, config
    ) -> None:
        super(WGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        training_parameters = config["training_parameters"]
        generator_config = config["generator_config"]
        discriminator_config = config["discriminator_config"]
        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)
        self.training_parameters = training_parameters
        self.kernel_size = training_parameters["kernel_size"]

    def training_step(self, batch: Tuple[torch.Tensor,torch.Tensor]):
        lr, hr = batch
        lr = lr.type(torch.FloatTensor).to(self.device)
        hr = hr.type(torch.FloatTensor).to(self.device)
        optimizer_g, optimizer_d = self.optimizers()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()


        # generator step
        self.toggle_optimizer(optimizer_g)
        sr = self.generator(lr)
        perceptual_loss = -self.discriminator(sr).mean()
        l1_loss = torch.nn.functional.l1_loss(hr, sr, reduction="mean")
        generator_loss = (
            self.training_parameters["coefficient_reconstruction_loss"] * l1_loss
            + self.training_parameters["coefficient_perceptual_loss"] * perceptual_loss
        )
        self.manual_backward(generator_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        self.log('generator_loss', generator_loss.item())
        self.log('l1_loss', l1_loss.item())


        # discriminator step
        self.toggle_optimizer(optimizer_d)
        for _ in range(self.training_parameters["discriminator_iterations_per_batch"]):
            sr = sr.detach()
            score_loss = (self.discriminator(sr) - self.discriminator(hr)).mean()
            gradients_interpolation = self.gradient_penalty(self.discriminator, hr, sr)
            lipschitz_loss = torch.linalg.norm(gradients_interpolation-1.0, dim=1).mean()
            discrminator_loss = score_loss + lipschitz_loss*self.training_parameters["coefficient_lipschitz_loss"]
        
            self.manual_backward(discrminator_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        if self.training_parameters["discriminator_iterations_per_batch"] > 0:
            self.log('discrminator_loss', discrminator_loss.item())
            self.log('score_loss', score_loss.item())


    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.training_parameters["lr_g"], betas=self.training_parameters["betas"])
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.training_parameters["lr_d"], betas=self.training_parameters["betas"])

        return [optimizer_g, optimizer_d], []

    def gradient_penalty(self,module, hr, sr):        
        epsilon = torch.rand(hr.shape[0], device=hr.device).view((-1,1,1,1,1)) 

        random_interpolation = hr*epsilon  + sr * (1-epsilon)
        random_interpolation = torch.autograd.Variable(random_interpolation, requires_grad=True)
        # random_interpolation.requires_grad = True
        score_interpolation = module(random_interpolation).sum()

        gradients_interpolation = torch.autograd.grad(
            score_interpolation,
            random_interpolation,
            create_graph=True,
            retain_graph=True, 
            allow_unused=True)[0]


        gradients_interpolation = gradients_interpolation.view((gradients_interpolation.shape[0], -1))

        return gradients_interpolation
    
    @torch.no_grad()
    def _superres(self, lr, step):

        kernel_size = self.kernel_size
        res = torch.zeros_like(lr)
        counts = torch.zeros_like(lr)

        dims = lr.shape[-3:]
        for i in tqdm(range(0,dims[0],step[0])):
            for j in range(0,dims[1],step[1]):
                for k in range(0,dims[2],step[2]):
                    block = lr[:,:,i:i+kernel_size[0], j:j+kernel_size[1], k:k+kernel_size[2]]
                    sr_block = self.generator(block)
                    res[:,:,i:i+kernel_size[0], j:j+kernel_size[1], k:k+kernel_size[2]] += sr_block
                    counts[:,:,i:i+kernel_size[0], j:j+kernel_size[1], k:k+kernel_size[2]]  += 1

        return res/counts


    def superres(self, in_path, out_path, step) -> None:
        lr = Nii_Handler.load(in_path)
        sr = self._superres(lr, step)
        Nii_Handler.save(sr, out_path)
