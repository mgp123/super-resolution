import math
import torch
from torch import nn
import pytorch_lightning as pl
from model.backbone_blocks import BackboneBlocks
from model.context_encoding import ContextEncoding
from model.self_attention import MultiHeadedAttentionSlow
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
        attention=False,
        context_encoding="lower_resolution",
        backbone_type="big_gan"
        ):
        super(Diffusion, self).__init__()
        self.save_hyperparameters()

        backbone_blocks = BackboneBlocks(backbone_type)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.scales = scales
        if context_encoding is not None:
            self.context_encoding = ContextEncoding(context_encoding)
        else:
            self.context_encoding = None

        self.initial_transform = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1)

        self.v_embedding = VarianceEmbedding(hidden_channels)

        in_blocks = []
        out_blocks = []

        current_channels = hidden_channels

        for s in (scales):
            if s > 1:
                in_blocks.append(
                    backbone_blocks.down(
                        in_channels= current_channels,
                        out_channels=current_channels*s,
                        embedding_size=hidden_channels
                        ))
                out_blocks.append(
                    backbone_blocks.up(
                        in_channels= current_channels*s*2,
                        out_channels=current_channels,
                        embedding_size=hidden_channels
                        ))
            elif s == 1:
                in_blocks.append(backbone_blocks.same(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    embedding_size=hidden_channels
                    ))
                out_blocks.append(backbone_blocks.same(
                    in_channels=current_channels*2,
                    out_channels=current_channels,
                    embedding_size=hidden_channels
                    ))
            else:
                raise ValueError(f"Can only handle scales bigger than 0. Recieved {s}")

            current_channels = s*current_channels



        out_blocks.reverse()

        self.in_blocks = nn.ModuleList(in_blocks)
        self.out_blocks = nn.ModuleList(out_blocks)

        if attention:
            self.attention = MultiHeadedAttentionSlow(
                in_channels=current_channels,
                out_channels=current_channels,
                value_channels=current_channels,
                n_heads=1,
                patch_size=1
                )
        else:
            self.attention = None
  

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

        if self.attention is not None:
            y = y + self.attention(y)

        for m, residual in zip(self.out_blocks, residuals):
            y = torch.cat([y,residual],dim=1)
            y = m(y, v_emv)
        
        return self.final_transform(y)

    def variance_scheudle(self,t,max_t=1000):
        s = 0.008
        base = math.cos(s*math.pi/ (2*(1+s)))
        t_value = torch.cos((t/max_t + s)*math.pi/ (2*(1+s)))

        return t_value/base

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=6e-5)
        return optimizer

    def reconstruction_loss(self, sr, hr):
        return torch.nn.functional.mse_loss(sr, hr)
    
    def get_variance(self, shape, max_t=1000, top_sample_step=1000):
        t = torch.randint(2, max_t+1, (shape[0],),dtype=torch.int32)
        variance = self.variance_scheudle(t, max_t)
        variance_prev = self.variance_scheudle(t-1, max_t)
        u = torch.rand(t.shape)
        variance = variance*u + variance_prev * (1-u)
        return variance

    def training_step(self, train_batch, batch_idx):
        hr, _ = train_batch
        hr = train_batch[0]
        hr = hr*2 - 1

        if self.context_encoding is not None:
            context = self.context_encoding(hr)

        variance = self.get_variance(hr.shape)
        variance = variance.to(hr.device)

        variance_unflat = variance.view(-1,1,1,1)


        noise = torch.randn_like(hr, device=hr.device)
        y = hr * torch.sqrt(variance_unflat) + noise * torch.sqrt(1-variance_unflat)

        if self.context_encoding is not None:
            y = torch.cat([y, context], dim=1)

        noise_predicion = self.forward(y, variance.view(-1,1))
        
        loss = self.reconstruction_loss(noise_predicion, noise)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        hr, _ = val_batch
        val_batch = val_batch[0]
        hr = hr*2 - 1

        if self.context_encoding is not None:
            context = self.context_encoding(hr)

        variance = self.get_variance(hr.shape)
        variance = variance.to(hr.device)

        variance_unflat = variance.view(-1,1,1,1)
        # variance = variance.to(val_batch.device)
        noise = torch.randn_like(val_batch)
        # print(variance.device, hr.device, noise.device)
        # print()
        # print()    
        y = hr * torch.sqrt(variance_unflat) + noise * torch.sqrt(1-variance_unflat)
        
        if self.context_encoding is not None:
            y = torch.cat([y, context], dim=1)
        
        noise_predicion = self.forward(y, variance.view(-1,1))
        
        loss = self.reconstruction_loss(noise_predicion, noise)
        self.log('val_loss', loss)
        return loss


# m = Diffusion(in_channels=32, hidden_channels=64,out_channels=3,scales=2)

# x = torch.randn((4,32,128,128))
# y = m(x)
# print(y.shape)
