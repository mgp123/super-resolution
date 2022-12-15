import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

from model.self_attention import MultiHeadedAttentionFast, MultiHeadedAttentionSlow

class PositionalEmbeddings():
    def __init__(self, period=1000):
        self.period = period

    def get(self, t, channels):
        # asumes channels is even 
        # if channels in self.mem:
        #     t_s = t.unsqueeze(-1)
        #     cos = torch.cos(t_s*self.mem[channels])
        #     sin = torch.sin(t_s*self.mem[channels])
        #     return torch.stack((cos,sin), dim=2).view(t_s.shape[0],-1)

        #     # return torch.cat([cos,sin])
        # else:
        #     half_channels = channels//2
        #     ks = torch.logspace( 
        #         start = 1.0/half_channels , 
        #         end = 1, 
        #         steps=half_channels,
        #         base= 1.0/self.period
        #         )
        #     self.mem[channels] = ks.unsqueeze(0).to(t.device)
        #     return self.get(t, channels)

        half_channels = channels//2
        ks = torch.logspace( 
            start = 1.0/half_channels , 
            end = 1, 
            steps=half_channels,
            base= 1.0/self.period,
            device=t.device
            )

        ks =  ks.unsqueeze(0)
        t_s = t.unsqueeze(-1)
        inds = ks * t_s
        cos = torch.cos(inds)
        sin = torch.sin(inds)
        return torch.stack((cos,sin), dim=2).view(t_s.shape[0],-1)

class PositionalEmbeddings2dGrid():
    def __init__(self, period=1000, save=True) -> None:
        self.pe = PositionalEmbeddings(period=period)
        self.mem = {}
        self.save = save

    def get(self, height, width, channels, device):
        half_channels = channels//2
        if (width, height, channels) in self.mem:
            return self.mem[width, height, channels].to(device)
        else:
            print(f"embedding w:{width}, h: {height}, c:{channels} not in save")
            w = torch.arange(width, device=device).repeat(height)
            w_embedding = self.pe.get(w,half_channels)
            w_embedding = w_embedding.view(1,height, width, half_channels).permute((0,3,1,2))
            h = torch.repeat_interleave(torch.arange(height,device=device),width)
            h_embedding = self.pe.get(h,half_channels)
            h_embedding = h_embedding.view(1,height, width, half_channels).permute((0,3,1,2))
            print(f"h_embedding  shape {h_embedding.shape}")
            space_embedding = torch.cat((w_embedding,h_embedding), dim=1)

            print(f"space embedding shape {space_embedding.shape}")

            if self.save:
                self.mem[width, height, channels] = space_embedding

            return space_embedding

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalization=True):
        super(ConvLayer, self).__init__()

        is_down = stride > 0
        stride = abs(stride)

        if is_down:
            convType = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=1,
                stride=stride
            )
        else:
            convType = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=1,
                output_padding=1 if stride == 2 else 0,
                stride=stride
            )

        self.model = nn.Sequential(
            convType,
            nn.GroupNorm(32,out_channels) if normalization else nn.Identity(),
            nn.LeakyReLU(),
        )


    def forward(self, x):
        return self.model(x)

class DownwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_long=True,  stride=2):
        super(DownwardBlock, self).__init__()

        self.is_long = is_long

        stride = -stride
        if is_long:
            self.long = ConvLayer(in_channels,out_channels) 
            stride = -stride
 
        self.right = ConvLayer(in_channels,out_channels, stride=stride) 
        self.down = ConvLayer(in_channels,in_channels)



    def forward(self, x):
        if self.is_long:
            return self.right(x), self.down(x), self.long(x)
        else:
            return self.right(x), self.down(x),


class RightwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_long=True, stride=2):
        super(RightwardBlock, self).__init__()

        self.is_long = is_long
        stride = -stride

        if is_long:
            self.long = ConvLayer(in_channels,out_channels) 
            stride = -stride

        self.right = ConvLayer(in_channels,out_channels, stride=stride) 


    def forward(self, x):
        if self.is_long:
            return self.right(x), self.long(x)
        else:
            return self.right(x)

class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_long=True, stride=2):
        super(DiffusionBlock, self).__init__()
        self.is_long = is_long
        self.rblock = RightwardBlock(in_channels,out_channels,is_long, stride) 
        self.dblock = DownwardBlock(in_channels,out_channels,is_long, stride) 
        self.v_emb1 = nn.Sequential( nn.Linear(1, 256), nn.SiLU(), nn.Linear(256,in_channels)) # let the net figure out the representation?
        self.v_emb2 = nn.Sequential( nn.Linear(1, 256), nn.SiLU(), nn.Linear(256,in_channels)) 

    def forward(self, x1, x2, variance):
        v1 = self.v_emb1(variance)
        # v1 = v1.view(v1.shape[0], v1.shape[1],1,1)
        v1.unsqueeze_(-1).unsqueeze_(-1)
        x1.add_(v1)

        v2 = self.v_emb2(variance)
        # v2 = v2.view(v2.shape[0], v2.shape[1],1,1)
        v2.unsqueeze_(-1).unsqueeze_(-1)
        x2.add_(v2)

        if self.is_long:
        # with profiler.record_function("is_long diffusion block part"):
            r1, d1 , l1 = self.dblock(x1)
            x2.add_(d1)
            r2, l2 = self.rblock(x2)
            return r1, r2, l1, l2
        else:
        # with profiler.record_function("is not long diffusion block part"):

            r1, d1  = self.dblock(x1)
            r2 = self.rblock(d1+x2)
            return r1, r2


class Diffusion(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_scales=3, channel_multiplier=2, attention_type="fast"):
        super(Diffusion, self).__init__()
        self.out_channels = out_channels
        self.init_transform1 = ConvLayer(in_channels,hidden_channels) 
        self.init_transform2 = ConvLayer(in_channels,hidden_channels)
        self.end_transform = nn.Sequential(
            ConvLayer(hidden_channels,hidden_channels, normalization=False),
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
 
        self.pe2d = PositionalEmbeddings2dGrid(period=100)

        if type(channel_multiplier) == int:
            channel_multiplier = [channel_multiplier]*n_scales

        mid_channels = hidden_channels
        for c in channel_multiplier:
            mid_channels *= c

        
        if attention_type == "fast":
            self.attention = MultiHeadedAttentionFast(in_channels=(mid_channels), value_channels=16, out_channels=mid_channels, n_heads=2)
        elif attention_type == "slow":
            self.attention = MultiHeadedAttentionSlow(in_channels=(mid_channels), value_channels=16, out_channels=mid_channels, n_heads=2)
        else:
            raise ValueError("attention type must be slow or fast")


        downscale_blocks = []
        upscale_blocks = []

        i_channels = hidden_channels

        for i in range(n_scales):
            downscale_blocks.append(
                DiffusionBlock(
                    in_channels=i_channels,
                    out_channels=i_channels*channel_multiplier[i] if i < (n_scales) else i_channels,
                    is_long=True,
                    stride=2 if i < n_scales -1 else 1,
                    ))
            upscale_blocks.append(
                DiffusionBlock(
                    in_channels=i_channels*channel_multiplier[i] if i < (n_scales) else i_channels,
                    out_channels=i_channels,
                    stride=2 if i > 0 else 1,
                    is_long=False,
                    )
                )

            i_channels = i_channels*channel_multiplier[i]

        upscale_blocks.reverse()
        self.upscale_blocks = nn.ModuleList(upscale_blocks)
        self.downscale_blocks = nn.ModuleList(downscale_blocks)

    def forward(self, x, variance):
    # with profiler.record_function("Init transform"):

        l1 = []
        l2 = []
        r1 = self.init_transform1(x)
        r2 = self.init_transform2(x)
    # with profiler.record_function("Down transform"):

        for k, downscale_block in enumerate(self.downscale_blocks):
            r1, r2, l1_current, l2_current = downscale_block(r1, r2, variance)
            l1.append(l1_current)
            l2.append(l2_current)
        
        l1.reverse()
        l2.reverse()

    # with profiler.record_function("attention transform"):

        # TODO check if attention is working
        B, C, H, W = r1.shape
        # space_embedding = self.pe2d.get(torch.arange(H*W),C) 
        # space_embedding = space_embedding.view(1,H, W, C).permute((0,3,1,2))
        # space_embedding = space_embedding.to(x.device)
        space_embedding = self.pe2d.get(H,W,C, r1.device) 

        # r1 = (self.attention(r1+space_embedding) )
        # r2 = (self.attention(r2+space_embedding) )

        r1.add_(self.attention(r1+space_embedding) )
        r2.add_(self.attention(r2+space_embedding) )


    # with profiler.record_function("up transform"):

        for upscale_block, l1_current, l2_current in zip(self.upscale_blocks, l1, l2):
            r1, r2 = upscale_block(r1+l1_current, r2+l2_current, variance)

        return self.end_transform(r1+r2)



# s = Diffusion(in_channels=64, hidden_channels=32)
# x = torch.randn((4,64,128,128))
# y = s(x, 0)
# print(y.shape)

