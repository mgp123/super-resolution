import sys
sys.path.append(sys.path[0] + "/..")
# print(sys.path)
import torch
import torch.autograd.profiler as profiler

from model.model import Diffusion
device = "cuda:0"
model = Diffusion(
    in_channels=6, 
    hidden_channels=128,
    out_channels=3,
    n_scales=3
    ).to(device)

x = torch.randn((4,6,32,32)).to(device)
variance = torch.ones((x.shape[0])).to(device).view(-1,1)
# y = s(x, 0)
# print(y.shape)
out = model(x, variance)


with profiler.profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
    for _ in range(4):
        out = model(x, variance)

print(prof.key_averages(group_by_stack_n=1).table(sort_by='cuda_memory_usage', row_limit=5))