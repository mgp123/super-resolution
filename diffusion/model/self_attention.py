import math
import torch
from torch import nn
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt

# Will the real slim shady please stand up?
class AttentionHead(nn.Module):
    def __init__(self, channels, value_channels, layer_norm=True):
        super(AttentionHead, self).__init__()
        self.init_transform = nn.Identity()

        if layer_norm:
            self.init_transform = nn.GroupNorm(num_groups=channels, num_channels=channels)

        self.query = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            )

        self.key = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            )

        self.value = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            )

        self.out_transform = nn.Conv2d(
            in_channels=value_channels, 
            out_channels=value_channels,
            kernel_size=1,
        )

        
    def forward(self, x):
        x = self.init_transform(x)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        B, D, H, W = K.shape
        # softmax(dot product)
        Q = Q.view(B,D,-1)
        Q = torch.transpose(Q, 1, 2)
        K = K.view(B,D,-1)
        # affinity = torch.zeros((B,H,W,H,W),device=Q.device)
        affinity = torch.bmm(Q, K)
        affinity = affinity/torch.sqrt(torch.tensor(D*1.0,device=K.device))
        # affinity = affinity.view(B, H, W, -1)
        affinity = torch.nn.functional.softmax(affinity,dim=2)
        # affinity = affinity.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


        V = V.view(B,D,-1)
        V = torch.transpose(V, 1, 2)

        attention = torch.bmm(affinity, V)
        # print("D", D, "H",H,"W",W)
        # print("attention", attention.shape)
        attention = torch.transpose(attention, 1, 2).view(B,D,H,W)

        # attention = torch.einsum("b i j k l , b d k l -> b d i j", affinity, V)
        # attention = torch.zeros((B,D,H,W),device=Q.device)
        # TODO should it be divided?
        # attention = attention/(H*W)

        return self.out_transform(attention)


class MultiHeadedAttentionSlow(nn.Module):
    def __init__(self, in_channels, value_channels, out_channels, n_heads, patch_size=1, learned_embedding=False):
        super(MultiHeadedAttentionSlow, self).__init__()
        self.patch_size = patch_size
        self.embedding = None
        embedding_channels = 0

        if learned_embedding:
            embedding_channels = 6
            self.embedding_shape = 8
            self.embedding = torch.nn.Parameter(torch.randn(1,embedding_channels,self.embedding_shape,self.embedding_shape))


        self.encode = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=value_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.heads = nn.ModuleList([AttentionHead(
            channels=value_channels+embedding_channels,
            value_channels=value_channels,
            ) for _ in range(n_heads)])
        
        self.upsample = nn.Upsample(scale_factor=patch_size) 

        self.decode = nn.Conv2d(
            in_channels=value_channels*n_heads, 
            out_channels=out_channels,
            kernel_size=1,
            )
        
        
    def forward(self, x):

        y = self.encode(x)
        if self.embedding != None:
            B, D, H, W = y.shape
            # assume squares (H==W)
            emb = torch.nn.functional.interpolate(self.embedding,H)
            emb = emb.expand(B, -1,-1,-1)
            y = torch.cat([y,emb], dim=1)

        attentions = torch.cat([head(y) for head in self.heads], dim=1)
        y = self.decode(self.upsample(attentions))
        return y

class MultiHeadedAttentionFast(nn.Module):
    def __init__(self, in_channels, value_channels, out_channels, n_heads):
        super(MultiHeadedAttentionFast, self).__init__()
        self.n_heads = n_heads
        self.sqrt_bottleneck_channels = math.sqrt(value_channels)

        self.query = nn.Parameter(torch.empty(n_heads, value_channels, in_channels))
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

        self.key = nn.Parameter(torch.empty(n_heads, value_channels, in_channels))
        nn.init.kaiming_uniform_(self.key, a=math.sqrt(5))
        
        self.value = nn.Parameter(torch.empty(n_heads, value_channels, in_channels))
        nn.init.kaiming_uniform_(self.value, a=math.sqrt(5))

        self.decode = nn.Conv2d(
            in_channels=value_channels*n_heads, 
            out_channels=out_channels,
            kernel_size=1,
            bias=False 
            )
        
    def forward(self, x):
        B, D, H, W = x.shape

        Q = torch.einsum("h k c, b c i j -> b h k i j", self.query, x)
        K = torch.einsum("h k c, b c i j -> b h k i j", self.key, x)
        V = torch.einsum("h k c, b c i j -> b h k i j", self.value, x)

        affinity = torch.einsum("b h d i j , b h d k l -> b h i j k l", Q, K).view(B, self.n_heads, H, W, -1)
        affinity = affinity/self.sqrt_bottleneck_channels
        affinity = torch.nn.functional.softmax(affinity,dim=4).view(B, self.n_heads, H, W, H, W)

        attention = torch.einsum("b h i j k l , b h d k l -> b h d i j", affinity, V)
        attention = attention/(H*W)     
        # attention = attention.permute((0,3,4,1,2))   
        attention = attention.reshape((attention.shape[0],-1,*attention.shape[-2:]))
        return self.decode(attention)

