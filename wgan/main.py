import torch
from model.discriminator import Discriminator
from model.generator import Generator

spatial_size = 40
in_channels = 1

g = Generator(
    in_channels=in_channels,
    n_dense_blocks=8,
    layers_per_dense_block=4
)
d = Discriminator(in_channels=in_channels, spatial_size=spatial_size)

g = g.to("cuda:0")
d = d.to("cuda:0")
x = torch.randn((8,in_channels,spatial_size,spatial_size,spatial_size))
x = x.to("cuda:0")
pytorch_total_params_g = sum(p.numel() for p in g.parameters())
pytorch_total_params_d = sum(p.numel() for p in d.parameters())

print(pytorch_total_params_g, pytorch_total_params_d, pytorch_total_params_g+pytorch_total_params_d)
with torch.cuda.amp.autocast():
    y = g(x)
    r = d(y)
# torch.save(
#     {"g": g.state_dict(),
#     "d": d.state_dict(),
#     }, 
#     "pepe.model")
