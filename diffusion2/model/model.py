import math
import torch
from torch import nn
import pytorch_lightning as pl

from model.big_gan_residual import BigGanResidualDown, BigGanResidualSame, BigGanResidualUp

class VarianceEmbedding(pl.LightningModule):
    def __init__(self, embedding_shape):
        super(VarianceEmbedding, self).__init__()
        self.embedding_shape = embedding_shape

        flat_dim = 1
        for k in embedding_shape:
            flat_dim *= k

    
        self.model = nn.Sequential(
            nn.Linear(1,flat_dim),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.model(x)
        y = y.view(y.shape[0],1,*self.embedding_shape )
        return y


class Diffusion(pl.LightningModule):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        scales, 
        variance_embedding_shape=(64,64),
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
            kernel_size=1)

        in_blocks = []
        variance_embedding_in = []
        out_blocks = []
        variance_embedding_out = []

        current_channels = hidden_channels

        for _ in range(scales):
            in_blocks.append(BigGanResidualDown(current_channels))
            out_blocks.append(BigGanResidualUp(current_channels*2))
            current_channels = 2*current_channels

            variance_embedding_in.append(
                VarianceEmbedding(variance_embedding_shape)
            )
            variance_embedding_out.append(
                    VarianceEmbedding(variance_embedding_shape)
                )

        for _ in range(same_dimension_blocks):
            in_blocks.append(BigGanResidualSame(current_channels))
            out_blocks.append(BigGanResidualSame(current_channels))
        
            variance_embedding_in.append(
                    VarianceEmbedding(variance_embedding_shape)
                )
            variance_embedding_out.append(
                    VarianceEmbedding(variance_embedding_shape)
                )


        variance_embedding_out.reverse()
        out_blocks.reverse()

        self.in_blocks = nn.ModuleList(in_blocks)
        self.out_blocks = nn.ModuleList(out_blocks)

        self.variance_embedding_in = nn.ModuleList(variance_embedding_in)
        self.variance_embedding_out = nn.ModuleList(variance_embedding_out)

        self.final_transform = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x, v):
        residuals = []
        y = self.initial_transform(x)

        for m, v_f in zip(self.in_blocks,self.variance_embedding_in):
            v_emv = v_f(v)
            v_emv = torch.nn.functional.interpolate(v_emv, size=y.shape[-2:])
            y = m(y + v_emv*(2**(-0.5)))

            residuals.append(y)

        residuals.reverse()

        for m, residual, v_f in zip(self.out_blocks, residuals,self.variance_embedding_out):
            v_emv = v_f(v)
            v_emv = torch.nn.functional.interpolate(v_emv, size=y.shape[-2:])

            y = m(y + (residual + v_emv) *(2**(-0.5)))
        
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
