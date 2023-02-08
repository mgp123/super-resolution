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


def variance_scheudle(relative_t):
    # betas = torch.linspace(1e-4,2e-2,1000)
    # variances = torch.cumprod(1-betas, dim=0)
    # indexes = int(1000* relative_t) - 1
    # # print(indexes, relative_t)
    # return variances[indexes]

    # return torch.tensor(0.07)
    # return torch.clip(torch.tensor(2*(relative_t))**2,min=0, max=0.999)
    # return (torch.tensor(1-relative_t)*0.8 + 0.2)**0.5
    s = 0.008
    base = math.cos(s*math.pi/ (2*(1+s)))
    t_value = torch.cos((relative_t + s)*math.pi/ (2*(1+s)))

    return (torch.tensor((t_value/base)))
    return 0.8 + 0.15*(torch.tensor((t_value/base)))

ckpt_file_path = "./runs/diffusion_generator_swish/version_1/checkpoints/epoch=39-step=43720.ckpt"
# d = Diffusion(in_channels=6, out_channels=3, hidden_channels=32,scales=2)
d = Diffusion.load_from_checkpoint(ckpt_file_path)

d = d.to("cuda:0")
# d = d.eval()
max_t = 1000

imgs = []

# image = torch.randn((4,3,spatial_dim,spatial_dim)).to("cuda:0")
def denioise(shape, context=None, init_t=max_t, step=1, end=0):
    image = torch.randn(shape, device="cuda:0")
    # low_res_image = low_res_image*0.5 + 1
    # variance = variance_scheudle(init_t, max_t).to("cuda:0")
    quant_size = 1000
    betas_g = torch.linspace(3e-5,1e-2,1000)
    variances_g = torch.cumprod(1-betas_g, dim=0)
    # variances_g = variance_scheudle(torch.linspace(0,1,1000))
    # print(indexes, relative_t)
    # # proxy for noisy high res
    # image = low_res_image * torch.sqrt(variance) + torch.randn_like(low_res_image) * torch.sqrt(1-variance)

    toPil = ToPILImage()
    var_list = []
    with torch.no_grad():
        for t in tqdm(range(init_t, end, -step)):
            coef = 1
            t = max(t,1)
            relative_t =  1 -(init_t - t)/(init_t-end)
            relative_t_prev = max( 1 - (init_t - t + step)/(init_t-end),0.002)

            # variance = torch.tensor(float(input("get input ")))
            # variance_prev = variance
            variance =  variances_g[int(quant_size* relative_t) - 1].to("cuda:0").repeat(shape[0])
            variance_prev =  variances_g[int(quant_size* relative_t_prev) - 1].to("cuda:0").repeat(shape[0])

            # variance = variance_scheudle(relative_t).to("cuda:0").repeat(shape[0])
            # variance_prev = variance_scheudle(relative_t_prev).to("cuda:0").repeat(shape[0])
            if t == init_t:
                variance_prev = variance_prev*0

            if t-step <= end :
                print("last")
                variance_prev = variance_prev*0 + 1
                variance = variance*0 + 1
                coef = 0


            
            x = image

            if context is not None:
                x = torch.cat([image, context],dim=1)

            noise_estimate = d(x, torch.tensor(variance).view(-1,1).to("cuda:0"))
            # noise_estimate = torch.clip(noise_estimate, min=-2, max=2)

            variance = variance.view(*(-1, *(len(image.shape) - 1)* [1]))
            variance_prev = variance_prev.view(*(-1, *(len(image.shape) - 1)* [1]))

            beta = betas_g[int(quant_size* relative_t) - 1].to("cuda:0").repeat(shape[0]).view(variance.shape)
            # beta = torch.clip(1-variance/variance_prev,max=0.999)
            alpha = 1 - beta 

            denoised_1_pass = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)
            denoised_1_pass = torch.clip(denoised_1_pass, min=-1., max=1.)

            denoised_coef =  beta  * torch.sqrt(variance_prev)  / (1 - variance)
            step_coef = torch.sqrt(alpha) * (1. - variance_prev) / (1 - variance)
            
            if t % 1 == 0:
                image_grid = make_grid( (0.5 + 0.5* image.cpu()), nrow=3)
                imgs.append(toPil(image_grid))
                if False:
                    print(t)
                    variance = variance*0 + torch.tensor(float(input("get input ")))
                    variance_prev = variance
                    print(step_coef.view(-1)[0])
                    print(denoised_coef.view(-1)[0])
                    print(beta.view(-1)[0])
                    plt.imshow(toPil(image_grid))
                    plt.show()
            
            noise_mult =  torch.sqrt(beta * (1. - variance_prev) / (1 - variance) )  # max(1 -  1.* ((t-init_t)/(end-init_t))**4,0)
            if relative_t < False:
                # variance = variance*0 + 1
                plt.imshow(toPil(image_grid))
                plt.show()

            noise = torch.randn_like(image)
            # image = denoised_1_pass* torch.sqrt(variance_prev) 
            # image = image*step_coef + denoised_1_pass*denoised_coef
            image = denoised_1_pass * torch.sqrt(variance_prev)  +  torch.sqrt(1-variance_prev)*noise*coef
            # image = denoised_1_pass * torch.sqrt(variance_prev)  +  noise_mult*noise*coef

            # image = (image -  noise_estimate * (beta) * torch.sqrt(1.0/(1-variance)) )  * torch.sqrt(1.0/alpha)  
            # image = (image ) -  noise_estimate *  1 * torch.sqrt(1.0/(1-variance))   * torch.sqrt(1.0/alpha) 
            # image = image - beta * noise_estimate       

            # image = image - noise*beta  * (1 - variance_prev) / (1 - variance)
            # image = torch.clip(image, min=-1, max=1)
            # image = torch.clip(image, min=0, max=.9)
            # image = denoised_1_pass
            # image = image + noise*coef*noise_mult
            # image = image + noise*noise_mult
            # clipping to help remove some of the estimate error for x_{t-1}
            clip_power = 2*torch.sqrt(1-variance_prev)
            # image = torch.clip(image, min=-1-clip_power, max=1+clip_power)
            var_list.append( ( variance.view(-1)[0]).cpu())
            # print(variance_prev)


    denoised = image
    # print(variance)

    # for k in [-1]:
    #     img = imgs[k]
    #     plt.imshow(img)
    #     plt.show()

    plt.plot(var_list)
    plt.show()

    img = imgs[-1]

    img.save(fp="out2.gif", format='GIF', append_images=imgs,
         save_all=True, duration=6, loop=0)
    # noise_estimate = d(torch.cat([image, low_res_image],dim=1), torch.tensor(variance_scheudle(1,max_t)).view(1,1).to("cuda:0"))
    # denoised = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)

    return image



data_loader_train, data_loader_test = get_data_loaders(4, 256)
# path = "local/example1.png"
# sample_image = torchvision.io.read_image(path)/255

sample_image, _ = next(iter(data_loader_test))
sample_image  = sample_image[0]
# sample_image = (sample_image*2) - 1
for spatial_dim, low_dim in [(64,32)]:
    print(spatial_dim)
    print(torch.min(sample_image))
    print(torch.max(sample_image))
    hr = transforms.Resize(spatial_dim)(sample_image).to("cuda:0")


    if d.context_encoding is not None:
        sample_image = d.context_encoding(sample_image)

        sample_image.unsqueeze_(0)
        sample_image = sample_image.to("cuda:0")
        sr = denioise((1,*hr.shape), context=sample_image, step=5, init_t=999, end=0).to("cuda:0")
        images = torch.stack([hr,sample_image[0], sr])

    else:
        sr =  0.5 + 0.5*denioise((3*5,*hr.shape), context=None, step=15, init_t=1000, end=0).to("cuda:0")
        images = sr

    images = 0.5*(images.cpu()+1)
    toPil = ToPILImage()

    image_grid = make_grid(images, nrow=3)
    print(images.shape)
    print(image_grid.shape)
    torchvision.utils.save_image(image_grid*2-1,f"sr2_{spatial_dim}.png")
    # plt.imshow(toPil(image_grid))
    # plt.savefig(f"sr2_{spatial_dim}.png")

    sample_image = sr.cpu()