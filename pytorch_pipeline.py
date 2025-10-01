import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import kagglehub

#path = kagglehub.dataset_download("hojjatk/mnist-dataset")
#print(f"Dataset downloaded to: {path}")

ROOT = "./kagglehub/datasets"

# define a simple transform to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# load the MNIST dataset
train_dataset = datasets.MNIST(root=ROOT, train=True, download=False, transform=transform)


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

# Now, apply normalization using computed mean and std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

train_dataset_norm = datasets.MNIST(root=ROOT, train=True, download=False, transform=transform_norm)
train_loader_norm = torch.utils.data.DataLoader(train_dataset_norm, batch_size=64, shuffle=True, num_workers=2)

# Split validation set from training set (e.g., 10% for validation)
val_size = int(0.1 * len(train_dataset_norm))
train_size = len(train_dataset_norm) - val_size
train_dataset_norm, val_dataset_norm = torch.utils.data.random_split(train_dataset_norm, (train_size, val_size))
train_loader_norm = torch.utils.data.DataLoader(train_dataset_norm, batch_size=64, shuffle=True)
val_loader_norm = torch.utils.data.DataLoader(val_dataset_norm, batch_size=64, shuffle=False)

# Load normalized test set
test_dataset_norm = datasets.MNIST(root=ROOT, train=False, download=False, transform=transform_norm)
test_loader_norm = torch.utils.data.DataLoader(test_dataset_norm, batch_size=64, shuffle=False)


