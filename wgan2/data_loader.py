from typing import Tuple
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils import data
from os.path import exists
import torch.fft
import cv2
import random
import numpy as np
import os
import glob
import SimpleITK as sitk


class BrainDataset(Dataset):
    def __init__(self, path, slice_size, low_size, passes=50, ) -> None:
        super(BrainDataset).__init__()

        self.files = glob.glob(path + "/**/*.nii*", recursive=True)*passes
        self.slice_size = slice_size
        self.low_size = low_size

    def __getitem__(self,index):
        itkimage = sitk.ReadImage(self.files[index])
        numpyImage = sitk.GetArrayFromImage(itkimage)*1.0
        numpyImage = torch.from_numpy(numpyImage)

        # normalization sort of
        numpyImage = (numpyImage - torch.mean(numpyImage)) / torch.std(numpyImage) 
        numpyImage = numpyImage.unsqueeze(0).unsqueeze(0)

        lr_numpyImage = torch.nn.functional.interpolate(
                numpyImage, size= (self.low_size)
            )

        lr_numpyImage = torch.nn.functional.interpolate(
            lr_numpyImage,
            size=(numpyImage.shape[2:])
        )


        numpyImage = numpyImage.squeeze(0).squeeze(0)
        lr_numpyImage = lr_numpyImage.squeeze(0).squeeze(0)


        indexes = []
        if self.slice_size is not None:
            for k, s in enumerate(self.slice_size):
                indexes.append(random.randint(0,numpyImage.shape[k]- s)) 
                
            res =  numpyImage[ indexes[0]:indexes[0]+self.slice_size[0], indexes[1]:indexes[1]+self.slice_size[1], indexes[2]:indexes[2]+self.slice_size[2]]
            res_lr =  lr_numpyImage[ indexes[0]:indexes[0]+self.slice_size[0], indexes[1]:indexes[1]+self.slice_size[1], indexes[2]:indexes[2]+self.slice_size[2]]
            res = res[np.newaxis,...]*1.0
            res_lr = res_lr[np.newaxis,...]*1.0
        else:
            return lr_numpyImage, numpyImage

        return res_lr, res

    def __len__(self):
        return len(self.files)
