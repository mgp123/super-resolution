from model.model import PositionalEmbeddings
import torch


t = torch.tensor(100).cpu()
pe = PositionalEmbeddings()
t1  = pe.get(t, 512)
print(t1.shape)
# print(t1)