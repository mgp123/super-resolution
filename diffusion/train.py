from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import math
from model.model import Diffusion
import torchvision

def freeze_network(m):
    for p in m.parameters():
        p.requires_grad = False 

def unfreeze_network(m):
    for p in m.parameters():
        p.requires_grad = True

def variance_scheudle(t, max_t):
    s = 0.01
    base = math.cos(s*math.pi/ (2*(1+s)))
    t_value = torch.cos((t/max_t + s)*math.pi/ (2*(1+s)))

    return t_value/base


def train():
    batch_size = 16
    in_channels = 3
    max_t = 1000
    learning_rate = 1e-5
    initial_epoch = 0
    epochs = 1
    epochs_per_checkpoint = 1
    write_loss_every = batch_size*4
    seen_samples = 0
    device = "cuda:0"

    model_name = "sr_2_scales_128_channels_fc_embeddings_16_dim_positional_attention_32_to_64_faster_attempts"
    makedirs("runs", exist_ok=True)
    makedirs("saved_weights", exist_ok=True)

    spatial_dim = 64
    downsize = torchvision.transforms.Resize(32)
    upsize = torchvision.transforms.Resize(spatial_dim)

    writer = SummaryWriter(log_dir="runs/" + model_name)
    scaler = torch.cuda.amp.GradScaler()

    d = Diffusion(
        in_channels=in_channels*2,
        hidden_channels=128,
        out_channels = in_channels,
        n_scales=2,
        attention_type="slow"
    )
    d = d.to(device)

    optimizer_d = torch.optim.Adam(
        d.parameters(),
        learning_rate,
    )
    

    if exists("saved_weights/" + model_name + ".model"):
        save = torch.load("saved_weights/" + model_name + ".model")
        d.load_state_dict(save['diffusion_dict'])
        optimizer_d.load_state_dict(save['diffusion_optimizer'])
        del save
        print("loaded from checkpoint")

    loss_getter = torch.nn.MSELoss(reduction="sum")


    data_loader_train, data_loader_test = get_data_loaders(batch_size, spatial_dim)
    variance_scheudle_progressive_alpha = 1*max_t/len(data_loader_train)
    print(len(data_loader_train))
    t_cap2 = 1

    for epoch in tqdm(range(initial_epoch, epochs), initial=initial_epoch, total=epochs, desc="epoch"):
        # for samples, _ in data_loader_train:

        for samples, _ in tqdm(data_loader_train, leave=False, desc="batch"):
            samples = (samples*2) - 1
            samples = samples.to(device)
            low_res = upsize(downsize(samples))

            
            t_cap = int(max(1, min(max_t-1, variance_scheudle_progressive_alpha* (seen_samples//batch_size) )))
            # t_cap = max_t-1
            # if t_cap2 != t_cap:
            #     print(t_cap)
            #     t_cap2 = t_cap


            t = torch.randint(1, t_cap+1, (samples.shape[0],))
            optimizer_d.zero_grad()

            with torch.cuda.amp.autocast():
                variance = variance_scheudle(t, max_t).view(-1,1,1,1).to(device)
                noise = torch.randn_like(samples)
                noisy_image = samples * torch.sqrt(variance) + noise * torch.sqrt(1-variance)

                y = d(torch.cat([noisy_image, low_res], dim=1) , variance.view(-1,1))
                loss = loss_getter(y,noise)/samples.shape[0]
            
           
            if torch.isnan(loss).any():
                raise ValueError('Found NaN during training')

            scaler.scale(loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            if seen_samples % write_loss_every == 0:
                writer.add_scalar("loss/samples", loss.item(), seen_samples)

            seen_samples += batch_size # inacuarate for end batch but keeps things simple

        if (epoch+1) % epochs_per_checkpoint == 0:
            torch.save(
            {"diffusion_dict": d.state_dict(),
            "diffusion_optimizer": optimizer_d.state_dict(),
            }, 
            "saved_weights/" + model_name + ".model"
            )

train()