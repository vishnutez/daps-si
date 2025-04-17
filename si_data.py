from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path
from torch.utils.data import Dataset
import warnings
from collections import defaultdict


__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            if __DATASET__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DATASET__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)


class SIDiffusionData(ABC, Dataset):
    def __init__(self, num_particles):
        self.num_particles = num_particles

    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_data(self, size=16, sigma=0):
        data = torch.stack([self.__getitem__(i) for i in range(size) for _ in range(self.num_particles)], dim=0)
        return data + torch.randn_like(data) * sigma

    def get_random(self, size=16, sigma=0):
        shape = (size, *self.get_shape())
        return torch.randn(shape) * sigma


@register_dataset('image')
class ImageDataset(SIDiffusionData):
    """
        A concrete class for handling image datasets, inherits from DiffusionData.

        This class is responsible for loading images from a specified directory,
        applying transformations to center crop the squared images of given resolution.

        Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        Output data range   : [-1, 1]
    """

    def __init__(self, num_particles, root='dataset/demo', resolution=256, device='cuda', start_id=None, end_id=None):
        # Define the file extensions to search for
        super().__init__(num_particles)
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(self.data)

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def __getitem__(self, i):
        img = (self.trans(Image.open(self.data[i])) * 2 - 1).to(self.device)
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)
        return img

    def get_shape(self):
        return 3, self.res, self.res

    def __len__(self):
        return len(self.data)
