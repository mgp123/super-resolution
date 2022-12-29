import math
import torch
from torch import nn
import pytorch_lightning as pl
from model.residual_block import Block, ResidualBlock, ResidualBlockDown, ResidualBlockUp

from model.big_gan_residual import BigGanResidualDown, BigGanResidualSame, BigGanResidualUp

class VarianceEmbedding(pl.LightningModule):
    def __init__(self, embedding_size):
        super(VarianceEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.model = nn.Sequential(
            nn.Linear(1,embedding_size*4),
            nn.SiLU(),
            nn.Linear(embedding_size*4,embedding_size),

        )
    def forward(self, x):
        return self.model(x)


class Diffusion(pl.LightningModule):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        scales, 
        same_dimension_blocks = 0):
        super(Diffusion, self).__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.scales = scales

        self.initial_transform = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1)

        self.v_embedding = VarianceEmbedding(hidden_channels)

        in_blocks = []
        out_blocks = []

        current_channels = hidden_channels

        for _ in range(scales):
            in_blocks.append(
                ResidualBlockDown(
                    in_channels= current_channels,
                    out_channels=current_channels*2,
                    embedding_size=hidden_channels
                    ))
            out_blocks.append(
                ResidualBlockUp(
                    in_channels= current_channels*2*2,
                    out_channels=current_channels,
                    embedding_size=hidden_channels
                    ))

            current_channels = 2*current_channels


        for _ in range(same_dimension_blocks):
            in_blocks.append(ResidualBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                embedding_size=hidden_channels
                ))
            out_blocks.append(ResidualBlock(
                in_channels=current_channels*2,
                out_channels=current_channels,
                embedding_size=hidden_channels
                ))

        out_blocks.reverse()

        self.in_blocks = nn.ModuleList(in_blocks)
        self.out_blocks = nn.ModuleList(out_blocks)


        self.final_transform = Block(
            in_channels=hidden_channels, 
            out_channels=out_channels,
            )

    def forward(self, x, v):
        residuals = []
        y = self.initial_transform(x)
        v_emv = self.v_embedding(v)


        for m in (self.in_blocks):
            y = m(y, v_emv)

            residuals.append(y)

        residuals.reverse()

        for m, residual in zip(self.out_blocks, residuals):
            y = torch.cat([y,residual],dim=1)
            y = m(y, v_emv)
        
        return self.final_transform(y)

    def variance_scheudle(self,t,max_t=1000):
        s = 0.01
        base = math.cos(s*math.pi/ (2*(1+s)))
        t_value = torch.cos((t/max_t + s)*math.pi/ (2*(1+s)))

        return t_value/base

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer

    def mse_loss(self, sr, hr):
        return torch.nn.functional.mse_loss(sr, hr)
    
    def get_variance(self, shape, max_t=1000):
        t = torch.randint(2, max_t+1, (shape[0],),dtype=torch.int32)
        variance = self.variance_scheudle(t, max_t)
        variance_prev = self.variance_scheudle(t-1, max_t)
        u = torch.rand(t.shape)
        variance = variance*u + variance_prev * (1-u)
        return variance

    def training_step(self, train_batch, batch_idx):
        hr, _ = train_batch
        hr = train_batch[0]
        lr = torch.nn.functional.interpolate(hr, scale_factor=0.5)
        lr = torch.nn.functional.interpolate(lr, scale_factor=2)
        
        variance = self.get_variance(lr.shape)
        variance = variance.to(hr.device)

        variance_unflat = variance.view(-1,1,1,1)


        noise = torch.randn_like(hr, device=hr.device)
        noisy_image = hr * torch.sqrt(variance_unflat) + noise * torch.sqrt(1-variance_unflat)

        y = torch.cat([noisy_image, lr], dim=1)
        noise_predicion = self.forward(y, variance.view(-1,1))
        
        loss = self.mse_loss(noise_predicion, noise)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        hr, _ = val_batch
        val_batch = val_batch[0]
        lr = torch.nn.functional.interpolate(hr, scale_factor=0.5)
        lr = torch.nn.functional.interpolate(lr, scale_factor=2)
        
        variance = self.get_variance(lr.shape)
        variance = variance.to(hr.device)

        variance_unflat = variance.view(-1,1,1,1)
        # variance = variance.to(val_batch.device)
        noise = torch.randn_like(val_batch)
        # print(variance.device, hr.device, noise.device)
        # print()
        # print()    
        noisy_image = hr * torch.sqrt(variance_unflat) + noise * torch.sqrt(1-variance_unflat)

        y = torch.cat([noisy_image, lr], dim=1)
        noise_predicion = self.forward(y, variance.view(-1,1))
        
        loss = self.mse_loss(noise_predicion, noise)
        self.log('val_loss', loss)
        return loss


# m = Diffusion(in_channels=32, hidden_channels=64,out_channels=3,scales=2)

# x = torch.randn((4,32,128,128))
# y = m(x)
# print(y.shape)
