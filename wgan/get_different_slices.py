from glob import glob
from os import makedirs
from os.path import exists
import torch
from tqdm import tqdm
from data_loader import BrainDataset, get_data_loaders
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
from torch.utils.data import Dataset

import SimpleITK as sitk


class RawBrainDataset(Dataset):
    def __init__(self, path,  passes=1, short_dim=160) -> None:
        super(BrainDataset).__init__()
        self.files = glob(path + "/**/*.nii*", recursive=True)*passes
        self.short_dim = short_dim

    def __getitem__(self,index):
        itkimage = sitk.ReadImage(self.files[-index])
        numpyImage = sitk.GetArrayFromImage(itkimage)*1.0
        numpyImage = torch.from_numpy(numpyImage)
        numpyImage = (numpyImage - torch.mean(numpyImage)) / torch.std(numpyImage) 


        numpyImage = numpyImage.unsqueeze(0)

        numpyImage = torch.nn.functional.interpolate(
                numpyImage, size= (256,256)
            )
        numpyImage = torch.nn.functional.interpolate(
                numpyImage, size= (256,256)
            )
        numpyImage = numpyImage.squeeze(0)

        return numpyImage
    
    def __len__(self):
        return len(self.files)

imgs = []

for short_dim in [1,2,3,4,5]:
    dataset = RawBrainDataset("local")
    # data_loader_train, data_loader_test = get_data_loaders(1, 3, slice_size=128, random_crop=False, dataset=dataset)
    # sample_image = torchvision.io.read_image(path)/255

    sample_image_iter = (iter(dataset))
    lr = next(sample_image_iter)
    lr = lr[::short_dim,:,:]
    lr = lr.repeat_interleave(short_dim, dim=0)
    print(lr.shape)

    # images = 0.5*(images.cpu()+1)

    result_image = sitk.GetImageFromArray(lr)
    sitk.WriteImage(result_image, f'slices_cuts/hr_short_as_{short_dim}.nii.gz')
