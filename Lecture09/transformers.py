'''
Complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

                               -- collected from original repository
'''

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./Dataset/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]  # n_features
        self.y = xy[:, [0]] # n_samples
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs = inputs * self.factor
        return inputs, targets

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(f'Features: {type(features)}, Labels: {type(labels)}')

composed = torchvision.transforms.Compose([
    ToTensor(),
    MulTransform(2.0)
])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(f'Features: {type(features)}, Labels: {type(labels)}')