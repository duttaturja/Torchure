import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./Dataset/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])  # n_features
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples
        self.n_samples = xy.shape[0]
        self.n_features = xy.shape[1] - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset = WineDataset()

# first_data = dataset[0]
# features, labels = first_data
# print(f'Features: {features}, Labels: {labels}')

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# data_iter = iter(dataloader)
# data = next(data_iter)
# features, labels = data
# print(f'Features: {features}, Labels: {labels}')

# training loop
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

def main():
    for epoch in range(2):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward pass 
            # backward pass
            # update weights
            if (i+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/2], Step [{i+1}/{n_iterations}], Inputs {inputs.shape}, Labels {labels.shape}')

train_dataset = torchvision.datasets.MNIST(root='./Dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
data = next(data_iter)
features, labels = data
print(f'Features: {features.shape}, Labels: {labels.shape}')


if __name__ == '__main__':
    # Your main script execution here
    main()
