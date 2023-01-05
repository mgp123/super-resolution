from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import math
from data_loaders import get_data_loaders
from model.model import Diffusion
from random import randrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.io
from torchvision.transforms import transforms


def variance_scheudle(t, max_t):
    s = 0.1
    base = math.cos(s*math.pi/ (2*(1+s)))
    t_value = math.cos((t/max_t + s)*math.pi/ (2*(1+s)))

    return torch.tensor((t_value/base))

ckpt_file_path = "runs/big_gan_diffusion/version_27/checkpoints/epoch=0-step=4374.ckpt"

# d = Diffusion(in_channels=6, out_channels=3, hidden_channels=32,scales=2)
d = Diffusion.load_from_checkpoint(ckpt_file_path)

d = d.to("cuda:0")
d = d.eval()
max_t = 1000

imgs = []

# image = torch.randn((4,3,spatial_dim,spatial_dim)).to("cuda:0")
def denioise(low_res_image, init_t=max_t, step=1, end=0):
    image = torch.randn_like(low_res_image)
    # low_res_image = low_res_image*0.5 + 1
    # variance = variance_scheudle(init_t, max_t).to("cuda:0")

    # # proxy for noisy high res
    # image = low_res_image * torch.sqrt(variance) + torch.randn_like(low_res_image) * torch.sqrt(1-variance)

    toPil = ToPILImage()
    with torch.no_grad():
        for t in tqdm(range(init_t, end, -step)):
            coef = 1
            t = max(t,1)
            variance = variance_scheudle(t, max_t).to("cuda:0")
            variance_prev = variance_scheudle(t-step, max_t).to("cuda:0")

            if t-step <= end - 1 :
                variance_prev = variance_prev*0 + 1
                coef = 0
            alpha = variance/variance_scheudle(max(t-1,1), max_t)
            beta = torch.clip(1 - alpha, max=0.99)
            
            noise_estimate = d(torch.cat([image, low_res_image],dim=1), torch.tensor(variance).view(1,1).to("cuda:0"))
            denoised_1_pass = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)
            image = (image -  noise_estimate * beta * torch.sqrt(1.0/(1-variance)) )  * torch.sqrt(1.0/alpha) 

            denoised_1_pass = torch.clip(denoised_1_pass, min=0, max=1)

            if t % 1 == 0:

                image_grid = make_grid((denoised_1_pass.cpu()), nrow=1)
                imgs.append(toPil(image_grid))


            noise = torch.randn_like(image)
            # image = image + torch.sqrt(1-variance_prev)*noise
            noise_mult = 0.0
            #image = denoised_1_pass* torch.sqrt(variance_prev)  + torch.sqrt(1-variance_prev)*noise*coef*noise_mult
            # image = torch.clip(image, min=0, max=.9)

            image = image + torch.sqrt(beta)*noise*coef*noise_mult

            # clipping to help remove some of the estimate error for x_{t-1}
            clip_power = torch.sqrt(variance_prev)  +  0.5*torch.sqrt(1-variance_prev) +  torch.sqrt(beta)*coef
            # image = torch.clip(image, min=-clip_power, max=1+clip_power)
            

    denoised = image

    # for k in [-1]:
    #     img = imgs[k]
    #     plt.imshow(img)
    #     plt.show()

    img = imgs[-1]

    img.save(fp="out2.gif", format='GIF', append_images=imgs,
         save_all=True, duration=6, loop=0)
    # noise_estimate = d(torch.cat([image, low_res_image],dim=1), torch.tensor(variance_scheudle(1,max_t)).view(1,1).to("cuda:0"))
    # denoised = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)

    return ToTensor()(img)



data_loader_train, data_loader_test = get_data_loaders(4, 256)
# path = "local/example1.png"
# sample_image = torchvision.io.read_image(path)/255

sample_image, _ = next(iter(data_loader_test))
sample_image  = sample_image[0]
# sample_image = (sample_image*2) - 1
for spatial_dim, low_dim in [(256,128)]:
    print(spatial_dim)
    print(torch.min(sample_image))
    print(torch.max(sample_image))
    hr = transforms.Resize(spatial_dim)(sample_image).to("cuda:0")

    sample_image = transforms.Resize(low_dim)(sample_image)
    sample_image = transforms.Resize(spatial_dim)(sample_image)

    sample_image.unsqueeze_(0)
    sample_image = sample_image.to("cuda:0")
    sr = denioise(sample_image, step=5, init_t=999, end=0).to("cuda:0")

    images = torch.stack([hr,sample_image[0], sr])

    images = 0.5*(images.cpu()+1)
    toPil = ToPILImage()

    image_grid = make_grid(images, nrow=3)
    torchvision.utils.save_image(image_grid*2-1,f"sr2_{spatial_dim}.png")
    # plt.imshow(toPil(image_grid))
    # plt.savefig(f"sr2_{spatial_dim}.png")

    sample_image = sr.cpu()