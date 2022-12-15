from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import math
from model.model import Diffusion
from random import randrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms import ToPILImage

def variance_scheudle(t, max_t):
    s = 0.1
    base = math.cos(s*math.pi/ (2*(1+s)))
    t_value = math.cos((t/max_t + s)*math.pi/ (2*(1+s)))

    return torch.tensor((t_value/base))

max_t = 1000
d = Diffusion(
    in_channels=3,
    hidden_channels=64,
    n_scales=3
)

path = "saved_weights/training_3_scales_64_channels_fc_embeddings_64_dim_positional_attention.model"
if exists(path):
    save = torch.load(path)
    d.load_state_dict(save['diffusion_dict'])
else:
    raise ValueError


d = d.to("cuda:0")
d = d.eval()

spatial_dim = 64


imgs = []

# image = torch.randn((4,3,spatial_dim,spatial_dim)).to("cuda:0")
def denioise(image, init_t=max_t, step=2):
    toPil = ToPILImage()
    with torch.no_grad():
        for t in tqdm(range(init_t, 0, -step)):
            noise = torch.randn_like(image)
            coef = 1
            variance = variance_scheudle(t, max_t).to("cuda:0")
            variance_prev = variance_scheudle(t-step, max_t).to("cuda:0")

            if t-step <= 0 :
                print("aaaaaaaaaaa ")
                variance_prev = variance_prev*0 + 1
                coef = 0
            alpha = variance/variance_scheudle(max(t-1,1), max_t)
            beta = 1- alpha #torch.clip(1 - alpha, max=0.999)
            # beta =  torch.clip(1- (variance/variance_scheudle(t-1,max_t)), max=0.999  )
            noise_estimate = d(image, torch.tensor(t).unsqueeze(0).to("cuda:0"))
            image = (image -  noise_estimate * beta * torch.sqrt(1.0/(1-variance)) )  * torch.sqrt(1.0/alpha) 
            # image = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)

            # image = (image -  noise_estimate  * torch.sqrt(1-variance))   * torch.sqrt(1.0/alpha) 
            image = torch.clip(image, min=-1, max=1)

            if t % 1 == 0:
                image_grid = make_grid(0.5*(image.cpu()+1), nrow=2)
                imgs.append(toPil(image_grid))
            # plt.imshow(image_grid.permute(1,2,0),)
            # plt.show()
            
            image = image  + torch.sqrt(beta)*torch.randn_like(image)*coef*1

            # image = image  * torch.sqrt(variance_prev) # + torch.randn_like(image)* torch.sqrt(1-variance_prev)
            # image = (image - noise_estimate * beta* torch.sqrt(1.0/(1-variance))  ) * (torch.sqrt(1.0/alpha))

    denoised = image
    # image = 0.5*(image.cpu() + 1)

    max_i = (torch.max(image))
    min_i = (torch.min(image))
    # print(max_i)
    # print(min_i)
    # plt.title("multi step reconstruction")

    # image = make_grid(image, nrow=2)

    img = imgs[0]

    img.save(fp="out.gif", format='GIF', append_images=imgs,
         save_all=True, duration=6, loop=0)

    return denoised


# plt.plot(alphas)
# plt.show()
def see_blurr():
    with torch.no_grad():

        data_loader_train, data_loader_test = get_data_loaders(1,spatial_size=spatial_dim)
        for samples, _ in tqdm(data_loader_test, leave=False, desc="batch"):
            samples = samples*2 - 1
            samples = samples.to("cuda:0")
            noise = torch.randn_like(samples)
            t_init = 600
            variance = variance_scheudle(t_init, max_t).to("cuda:0")
            print("noise boudns, ", torch.min(noise), torch.max(noise))
            print("variance sample noise", variance)

            noisy_image = samples * torch.sqrt(variance) + noise * torch.sqrt(1-variance)
            y = d(noisy_image , torch.tensor(t_init).to("cuda:0"))
            # denoised = denioise(torch.randn_like(noisy_image), init_t=t_init, step=1)
            denoised = denioise((noisy_image), init_t=t_init, step=10)

            denoised_1_pass = (noisy_image -  y * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)


            samples = 0.5*(samples.cpu()+1)
            noisy_image = 0.5*(noisy_image.cpu()+1)
            denoised_1_pass = 0.5*(denoised_1_pass.cpu() + 1)
            denoised = 0.5*(denoised.cpu() + 1)

            plt.title("original")

            plt.imshow(samples[0].permute(1,2,0),)
            plt.show()

            plt.title("noisy")

            plt.imshow(noisy_image[0].permute(1,2,0),)
            plt.show()

            plt.title("1 pass reconstruction")
            plt.imshow(denoised_1_pass[0].permute(1,2,0),)
            plt.show()

            plt.title("multi step reconstruction")
            plt.imshow(denoised[0].permute(1,2,0),)
            plt.show()

            max_i = (torch.max(denoised_1_pass))
            min_i = (torch.min(denoised_1_pass))
            print(max_i)
            print(min_i)



            break
see_blurr()