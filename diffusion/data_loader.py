from typing import Tuple
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils import data
from os.path import exists


# slightly modified version of the one used in another project 
def get_data_loaders(batch_size: int, spatial_size) -> Tuple[data.DataLoader, data.DataLoader]:
    t = transforms.Compose([transforms.Resize(spatial_size) ,transforms.ToTensor()])
    dataset = datasets.ImageFolder("dataset/", t)
    test_set_size = batch_size
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