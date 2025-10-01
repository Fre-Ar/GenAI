import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import kagglehub

#path = kagglehub.dataset_download("hojjatk/mnist-dataset")
#print(f"Dataset downloaded to: {path}")


# define a simple transform to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# load the MNIST dataset
train_dataset = datasets.MNIST(root="./kagglehub/datasets", train=True, download=False, transform=transform)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset), shuffle=False
)
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Compute mean and std
mean = images.mean().item()
std = images.std().item()

print(f"MNIST mean: {mean:.4f}")
print(f"MNIST std: {std:.4f}")

