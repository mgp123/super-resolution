from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from data_loader import BrainDataset, VideoDataset, generic_loaders, get_data_loaders, get_data_loaders_dummy, low_pass_filter
from model.discriminator import Discriminator
from model.generator import Generator
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import torchvision

class dummyWith():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass

def freeze_network(m):
    for p in m.parameters():
        p.requires_grad = False 

def unfreeze_network(m):
    for p in m.parameters():
        p.requires_grad = True

def mean_and_blurr(kernel_size=19):
    def res(x):
        y = torch.mean(x,dim=1)
        y = torchvision.transforms.functional.gaussian_blur(y,kernel_size)
        y = y.unsqueeze(1)
        return y
    return res

def get_low_resolution_method(**kwargs):

    low_size = kwargs.get("low_size", kwargs["spatial_size"][0]//2)
    # downsize = torchvision.transforms.Resize(low_size)
    # upsize = torchvision.transforms.Resize(kwargs["spatial_size"])
    
    def down_size(x):
        return torch.nn.functional.interpolate(
            torch.nn.functional.interpolate(
                x, size= low_size
            ),
            size=kwargs["spatial_size"]
        )
        # return upsize(downsize(x))

    return down_size

        
    def low_pass(x):
        return low_pass_filter(x, **kwargs)

def train():
    batch_size = 8
    spatial_size = (64,40,64)
    in_channels = 1
    out_channels=1
    learning_rate = 2e-4
    initial_epoch = 0
    epochs = 40
    epochs_per_checkpoint = 10
    write_loss_every = batch_size*5
    seen_samples = 0
    coefficient_perceptual_loss = 0.01
    coefficient_reconstruction_loss = 0.99
    coefficient_lipschitz_loss = 1e-2
    low_pass_filter_cut_bin = 5
    device = "cuda:0"
    discriminator_iterations_per_batch = 1
    model_name = "trainning_brain3"
    makedirs("runs", exist_ok=True)
    makedirs("saved_weights", exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/{model_name}")
    use_scaler = False
    opener = torch.cuda.amp.autocast if use_scaler else dummyWith
    scaler = torch.cuda.amp.GradScaler()
    dimension = 3

    g = Generator(
        dimension,
        in_channels=in_channels,
        out_channels=out_channels,
        n_dense_blocks=3,
        layers_per_dense_block=2
    )
    d = Discriminator(dimension,in_channels=out_channels, spatial_size=spatial_size)


    g = g.to(device)
    d = d.to(device)

    optimizer_g = torch.optim.Adam(
        g.parameters(),
        learning_rate,
    )
    optimizer_d = torch.optim.Adam(
        d.parameters(),
        learning_rate,
    )
    # low_resolution_method = mean_and_blurr(kernel_size=21)


    if exists(f"saved_weights/{model_name}.model"):
        save = torch.load(f"saved_weights/{model_name}.model")
        g.load_state_dict(save['generator_dict'])
        d.load_state_dict(save['discriminator_dict'])
        optimizer_d.load_state_dict(save['discriminator_optimizer'])
        optimizer_g.load_state_dict(save['generator_optimizer'])
        print("resumed training from save")

        del save


    

    video_paths = "local/scenes"
    crop_size = spatial_size[1]
    frames = spatial_size[0]
    # data_loader_train, data_loader_test = generic_loaders(
    #     VideoDataset(video_paths=video_paths,crop_size=crop_size,frames_size=frames),
    #      batch_size)

    dataset = BrainDataset("local", slice_size=spatial_size,  low_size=(256,32,128))
    low_resolution_method = get_low_resolution_method(spatial_size=spatial_size, low_size=(64,8,64))
    data_loader_train, data_loader_test = get_data_loaders(batch_size, dimension, spatial_size[0], random_crop=False, dataset=dataset)

    for epoch in tqdm(range(initial_epoch, epochs), initial=initial_epoch, total=epochs, desc="epoch"):
        for samples_hr, samples_lr in tqdm(data_loader_train, leave=False, desc="batch"):
            samples_hr = samples_hr.to(device,dtype=torch.float)
            samples_lr = samples_lr.to(device,dtype=torch.float)

            # samples_lr = low_resolution_method(samples_hr, cut_bin=low_pass_filter_cut_bin)


            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with opener():
                samples_sr = g(samples_lr)
                samples_sr_constant = samples_sr.detach()

            unfreeze_network(d)
            freeze_network(g)
            
            for _ in range(discriminator_iterations_per_batch):
                with opener():
                    score_loss = (d(samples_sr_constant) - d(samples_hr)).mean()
                    flattened_dims = [-1,1] + [1]*dimension
                    
                    epsilon = torch.rand(samples_hr.shape[0], device=device).view(*flattened_dims) 
                    random_interpolation = samples_hr*epsilon + samples_sr_constant * (1-epsilon)
                    # TODO should we detach? 
                    # If we are only using this as a random point to make the lipschitz loss it makes sense to detach, maybe?
                    # random_interpolation = random_interpolation.detach()
                    # curently no need to detach as there is no graph on random_interpolation
                    random_interpolation.requires_grad = True
                    # TODO check if this sum trick to calculate all gradients in one pass works
                    score_interpolation = d(random_interpolation).sum()

                    gradients_interpolation = torch.autograd.grad(
                        score_interpolation,
                        random_interpolation,
                        create_graph=True,
                        retain_graph=True, 
                        allow_unused=True)[0]


                    gradients_interpolation = gradients_interpolation.view((gradients_interpolation.shape[0], -1))
                    lipschitz_loss = torch.linalg.norm(gradients_interpolation-1.0, dim=1).mean()
                    discrminator_loss = score_loss + lipschitz_loss*coefficient_lipschitz_loss
                
                if torch.isnan(discrminator_loss).any():
                    raise ValueError('Found NaN during training')

                if use_scaler:
                    scaler.scale(discrminator_loss).backward()
                    scaler.step(optimizer_d)
                    scaler.update()
                else:
                    discrminator_loss.backward()
                    optimizer_d.step()

                optimizer_d.zero_grad()


            unfreeze_network(g)
            freeze_network(d)

            with opener():
                perceptual_loss = - d(samples_sr).mean()
                l1_loss = torch.nn.functional.l1_loss(samples_hr, samples_sr,reduction='mean')
                generator_loss =  coefficient_reconstruction_loss*l1_loss + coefficient_perceptual_loss * perceptual_loss

            if torch.isnan(generator_loss).any():
                raise ValueError('Found NaN during training')

            if use_scaler:
                scaler.scale(generator_loss).backward()
                scaler.step(optimizer_g)
                scaler.update()
            else:
                generator_loss.backward()
                optimizer_g.step()

            optimizer_g.zero_grad()

            if seen_samples % write_loss_every == 0:
                writer.add_scalar("score_loss/samples", score_loss.item(), seen_samples)
                writer.add_scalar("lipschitz_loss/samples", lipschitz_loss.item(), seen_samples)
                writer.add_scalar("discrminator_loss/samples", discrminator_loss.item(), seen_samples)
                writer.add_scalar("l1_loss/samples", l1_loss.item(), seen_samples)
                writer.add_scalar("perceptual_loss/samples", perceptual_loss.item(), seen_samples)
                writer.add_scalar("generator_loss/samples", generator_loss.item(), seen_samples)

            seen_samples += samples_hr.shape[0]

        if (epoch+1) % epochs_per_checkpoint == 0:
            torch.save(
            {"discriminator_dict": d.state_dict(),
            "generator_dict": g.state_dict(),
            "discriminator_optimizer": optimizer_d.state_dict(),
            "generator_optimizer": optimizer_g.state_dict(),
            }, 
            f"saved_weights/{model_name}.model"
            )

train()