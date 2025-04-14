import numpy as np
import os
import torch
from torch.utils.data import random_split
from torchvision import transforms
from train import train_resnet_transf_unet
from torch.utils.data import DataLoader
from data_loader import TrainImageDataset

num_epochs = 4
directory = 'ethz-cil-monocular-depth-estimation-2025/train/train/'
device = torch.device("cuda")

files = os.listdir(directory)
input_images_files = [directory+ file for file in files if file.endswith('.png')]
output_mask_files = [directory+ file for file in files if file.endswith('.npy')]

input_images_files = sorted(input_images_files)
output_mask_files = sorted(output_mask_files)

seed_generator = torch.Generator().manual_seed(32)

transform_rgb = transforms.Compose([
    transforms.ToTensor()
])

transforms_mask = transforms.Compose([
    transforms.ToTensor()
])

dataset = TrainImageDataset(input_images_files, output_mask_files,transform=transform_rgb, transform_mask=transforms_mask)
train_size, test_size = int(len(dataset) * 0.8), len(dataset) - (int(len(dataset) * 0.8))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=seed_generator)
dataloader_train = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False)
train_resnet_transf_unet(dataloader_train, device, num_epochs)