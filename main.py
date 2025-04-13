import numpy as np
import os
import torch
from torch.utils.data import random_split
from torchvision import transforms
from dataset_create import LazyImageDataset
from train import train_net
from test import test_net
import wandb

num_epochs = 4
directory = '/home/aryan-sood/Documents/CIL/ethz-cil-monocular-depth-estimation-2025/train/train/'
device = torch.device("cuda")

wandb.init(project="my-pytorch-training")
wandb.config = {
    "epochs": 5,
    "lr": 0.001,
    "batch_size": 64
}

files = os.listdir(directory)
input_images_files = [directory+ file for file in files if file.endswith('.png')]
output_mask_files = [directory+ file for file in files if file.endswith('.npy')]

input_images_files = sorted(input_images_files)
output_mask_files = sorted(output_mask_files)

generator = torch.Generator().manual_seed(42)

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = LazyImageDataset(input_images_files, output_mask_files,transform=transform)
train_size, test_size = int(len(dataset) * 0.8), len(dataset) - (int(len(dataset) * 0.8))
print(test_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
test_net(test_dataset, device, test_size)
#train_net(train_dataset, device, num_epochs, test_dataset, wandb)