from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import math
from model.generator import Generator
from random import randrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.io
from torchvision.transforms import transforms
import torch.nn.functional as F


def make_superres(g, lr, overlap, kernel_size=64):
    with torch.no_grad():

        # return g(lr.view(1,3,64,64))[0]
        W, H = lr.shape[1], lr.shape[2]
        sr = lr.unfold(dimension=1, size=kernel_size, step=kernel_size-overlap)
        sr = sr.unfold(dimension=2, size=kernel_size, step=kernel_size-overlap)

        n1, n2 = sr.shape[1], sr.shape[2]
        sr = sr.reshape(3,-1, kernel_size,kernel_size)
        sr = torch.transpose(sr, 0, 1)

        print("lr", lr.shape)

        print("sr", sr.shape)

        sr = g(sr)
        sr = torch.clamp(sr,min=0,max=1)
        #sr[0::2] = sr[0]*0
        #sr[:,:,32:,:] = 0
        #sr[:,:,32:,:] = 0

        sr = torch.transpose(sr, 0, 1)
        sr = sr.reshape(3,n1,n2, kernel_size,kernel_size)

        print("sr", sr.shape)

        # slow method but it doesnt matter because its done on inference
        res = torch.zeros(3,W,H, device=sr.device)
        counter = torch.zeros(3,W,H, device=sr.device)

        step = kernel_size-overlap

        for i in range(sr.shape[1]):
            for j in range(sr.shape[2]):
                ki = step * i
                kj = step * j
                res[:,ki:ki+kernel_size, kj:kj+kernel_size] += sr[:,i,j]
                counter[:,ki:ki+kernel_size, kj:kj+kernel_size] += 1
        
        res = res/counter

        print("sr", sr.shape)

    return res

# print(get_n_params(d))

model_path = "saved_weights/trainning_large.model"
spatial_size = 128
batch_size = 1
low_pass_filter_cut_bin = 3
g = Generator(
    dimensions=2,
    in_channels=3,
    out_channels=3,
    n_dense_blocks=4,
    layers_per_dense_block=4
)


if exists(model_path):
    save = torch.load(model_path)
    g.load_state_dict(save['generator_dict'])
else:
    raise ValueError


g = g.to("cuda:0")
g = g.eval()


imgs = []


data_loader_train, data_loader_test = get_data_loaders(4, 2, slice_size=128, random_crop=False)
# sample_image = torchvision.io.read_image(path)/255

sample_image, _ = next(iter(data_loader_test))
sample_image  = sample_image[0]

path = "local/faces/example3.png"
# sample_image = torchvision.io.read_image(path)/255

# sample_image = (sample_image*2) - 1
for spatial_dim, low_dim in [(128, 64)]:
    print(spatial_dim)
    print(torch.min(sample_image))
    print(torch.max(sample_image))
    hr = transforms.Resize(spatial_dim)(sample_image).to("cuda:0")

    sample_image = transforms.Resize(low_dim)(sample_image)
    sample_image = transforms.Resize(spatial_dim)(sample_image)

    # sample_image.unsqueeze_(0)
    sample_image = sample_image.to("cuda:0")
    sr = make_superres(g,sample_image, overlap=0,kernel_size=32)

    images = torch.stack([hr,sample_image, sr])

    # images = 0.5*(images.cpu()+1)
    images.cpu()
    toPil = ToPILImage()

    image_grid = make_grid(images, nrow=3)
    plt.imshow(toPil(image_grid))
    plt.savefig(f"sr2_{spatial_dim}.png")

    sample_image = sr.cpu()