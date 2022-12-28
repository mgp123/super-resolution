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
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.io
from torchvision.transforms import transforms


def variance_scheudle(t, max_t):
    s = 0.1
    base = math.cos(s*math.pi/ (2*(1+s)))
    t_value = math.cos((t/max_t + s)*math.pi/ (2*(1+s)))

    return torch.tensor((t_value/base))

max_t = 1000
d = Diffusion(
    in_channels=3*2,
    hidden_channels=128,
    out_channels = 3,
    n_scales=2,
    attention_type="slow"
)


# print(get_n_params(d))

# vs = [1-variance_scheudle(t,max_t) for t in range(max_t)]
# plt.plot(vs)
# plt.show()
# 1/0
model_name = "big_gan_res_sr_2_scales_64_channels_2_channel_multiplier_fc_embeddings_32_dim_positional_16_patch_attention_12_heads_64_to_128_none_attention"
path = f"saved_weights/{model_name}.model"
d = Diffusion.load(path)
# if exists(path):
#     save = torch.load(path)
#     d.load_state_dict(save['diffusion_dict'])
# else:
#     raise ValueError


d = d.to("cuda:0")
d = d.eval()


imgs = []

# image = torch.randn((4,3,spatial_dim,spatial_dim)).to("cuda:0")
def denioise(low_res_image, init_t=max_t, step=1, end=0):
    image = torch.randn_like(low_res_image)
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
            beta = torch.clip(1 - alpha, max=0.999)
            
            noise_estimate = d(torch.cat([image, low_res_image],dim=1), torch.tensor(variance).view(1,1).to("cuda:0"))
            denoised_1_pass = (image -  noise_estimate * torch.sqrt(1-variance) ) * torch.sqrt(1.0/variance)
            image = (image -  noise_estimate * beta * torch.sqrt(1.0/(1-variance)) )  * torch.sqrt(1.0/alpha) 

            denoised_1_pass = torch.clip(denoised_1_pass, min=-1, max=1)

            if t % 1 == 0:

                image_grid = make_grid(0.5*(denoised_1_pass.cpu()+1), nrow=1)
                imgs.append(toPil(image_grid))


            noise = torch.randn_like(image)
            # image = image + torch.sqrt(1-variance_prev)*noise
            noise_mult = 0.1
            # image = denoised_1_pass* torch.sqrt(variance_prev)  + torch.sqrt(1-variance_prev)*noise*coef*noise_mult
            image = image + torch.sqrt(beta)*noise*coef*noise_mult

            # clipping to help remove some of the estimate error for x_{t-1}
            clip_power = torch.sqrt(variance_prev)  +  2*torch.sqrt(1-variance_prev) +  torch.sqrt(beta)*coef
            
            # image = torch.clip(image, min=-clip_power, max=clip_power)

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

    return (ToTensor()(img)*2 - 1)



path = "local/example1.png"
data_loader_train, data_loader_test = get_data_loaders(1, 128)
sample_image = torchvision.io.read_image(path)/255

sample_image, _ = next(iter(data_loader_test))
sample_image  = sample_image[0]
sample_image = (sample_image*2) - 1
for spatial_dim, low_dim in [(256,128)]:
    print(spatial_dim)
    print(torch.min(sample_image))
    print(torch.max(sample_image))
    hr = transforms.Resize(spatial_dim)(sample_image).to("cuda:0")

    sample_image = transforms.Resize(low_dim)(sample_image)
    sample_image = transforms.Resize(spatial_dim)(sample_image)

    sample_image.unsqueeze_(0)
    sample_image = sample_image.to("cuda:0")
    sr = denioise(sample_image, step=2, init_t=999, end=0).to("cuda:0")

    images = torch.stack([hr,sample_image[0], sr])

    images = 0.5*(images.cpu()+1)
    toPil = ToPILImage()

    image_grid = make_grid(images, nrow=3)
    torchvision.utils.save_image(image_grid,f"sr2_{spatial_dim}.png")
    # plt.imshow(toPil(image_grid))
    # plt.savefig(f"sr2_{spatial_dim}.png")

    sample_image = sr.cpu()