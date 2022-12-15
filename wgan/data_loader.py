from typing import Tuple
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils import data
from os.path import exists
import torch.fft


def low_pass_filter(x: torch.Tensor, cut_bin=20):
    n = x.shape[-1]
    fftn = torch.fft.fftn(x, dim=[-3,-2,-1])

    maskX = (torch.arange(n**3, device=x.device).reshape(n,n,n) % n)
    mask = ( maskX <= cut_bin ) 
    mask =  mask * torch.transpose(mask, 0, 1) * torch.transpose(mask, 0, 2)
    fftn = fftn*mask

    return torch.fft.ifftn(fftn, dim=[-3,-2,-1]).real

# slightly modified version of the one used in another project 
def get_data_loaders(batch_size: int, dimension: int, slice_size: int = 40, random_crop:bool=True) -> Tuple[data.DataLoader, data.DataLoader]:
    if not exists("dataset"):
        raise Exception("No dataset found. You need to put your directory with the images inside the dataset "
                        "directory")
    t = transforms.ToTensor()
    if random_crop:                
        t = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.RandomCrop(size=(slice_size,)*dimension),

                #transforms.RandomCrop(size=(([slice_size]*dimension))),
            ]
            )


    dataset = datasets.ImageFolder("dataset", t)
    test_set_size = 4
    test_set, train_set = data.random_split(
        dataset,
        [test_set_size, len(dataset) - test_set_size],
        generator=torch.Generator().manual_seed(2022)
    )

    data_loader_train = data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    data_loader_test = data.DataLoader(
        test_set,
        batch_size=test_set_size,
        shuffle=False,
        pin_memory=True
    )

    return data_loader_train, data_loader_test


class RandomDataset(Dataset):
  def __init__(self, cube_size: int, dataset_size: int):
    super(RandomDataset, self).__init__()
    self.cube_size = cube_size
    self.dataset_size = dataset_size

    self.X = (torch.arange(cube_size**3) % cube_size**2)
    self.X = self.X/(cube_size**2) - 0.5

    self.Y = torch.arange(cube_size**3)%cube_size
    self.Y = self.Y/cube_size - 0.5

    self.Z = torch.arange(cube_size**3)
    self.Z = self.Z/(cube_size**3) - 0.5


  def __len__(self):
    return self.dataset_size

  def __getitem__(self, index):
    p = (torch.rand(3)-0.5)/2
    # p = p*0
    res = ((self.X-p[0])**2 + (2*self.Y-p[1])**2 + (torch.cos(self.Z-p[2]))  )
    # res = torch.cos(self.X*self.Y) 
    res = res.view(1,self.cube_size,self.cube_size,self.cube_size) 
    res = (res <= 1.1) 
    return 1.0 * res, 0 
    return res/torch.max(res), 0
    # return torch.rand((1,self.cube_size,self.cube_size,self.cube_size)) , 0


def get_data_loaders_dummy(batch_size: int, cube_size: int = 40) -> Tuple[data.DataLoader, data.DataLoader]:

    dataset = RandomDataset(cube_size=cube_size, dataset_size=128)
    test_set_size = 4
    test_set, train_set = data.random_split(
        dataset,
        [test_set_size, len(dataset) - test_set_size],
        generator=torch.Generator().manual_seed(2022)
    )

    data_loader_train = data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    data_loader_test = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return data_loader_train, data_loader_test