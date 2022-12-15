import math
import torch
from torch import nn
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt

# Will the real slim shady please stand up?
class AttentionHead(nn.Module):
    def __init__(self, channels, value_channels):
        super(AttentionHead, self).__init__()

        self.query = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            bias=False 
            )

        self.key = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            bias=False 
            )

        self.value = nn.Conv2d(
            in_channels=channels, 
            out_channels=value_channels,
            kernel_size=1,
            bias=False 
            )

        
    def forward(self, x):
        B, D, H, W = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # softmax(dot product)
        affinity = torch.einsum("b d i j , b d k l -> b i j k l", Q, K).view(B, H, W, -1)
        affinity = torch.nn.functional.softmax(affinity,dim=3).view(B, H, W, H, W)

        attention = torch.einsum("b i j k l , b d k l -> b d i j", affinity, V)
        # TODO should it be divided?
        # attention = attention/(H*W)

        return attention


class MultiHeadedAttentionSlow(nn.Module):
    def __init__(self, in_channels, value_channels, out_channels, n_heads):
        super(MultiHeadedAttentionSlow, self).__init__()

        self.heads = nn.ModuleList([AttentionHead(
            channels=in_channels,
            value_channels=value_channels
            ) for _ in range(n_heads)])
        

        self.decode = nn.Conv2d(
            in_channels=value_channels*n_heads, 
            out_channels=out_channels,
            kernel_size=1,
            bias=False 
            )
        
        
    def forward(self, x):
        attentions = torch.cat([head(x) for head in self.heads], dim=1)
        y = self.decode(attentions)
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

