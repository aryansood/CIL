import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from model import UNet
import matplotlib.pyplot as plt
import albumentations as albume
import cv2

class LazyImageDataset(Dataset):
    def __init__(self, image_paths, mask_path,transform=None, transform_numpy = None):
        self.image_paths = image_paths
        self.mask_path = mask_path
        self.transform = transform
        self.transform_numpy = transform_numpy

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_pat = self.mask_path[idx]
        to_tensor = transforms.ToTensor()
        image = Image.open(img_path).convert("RGB")
        mask = np.load(mask_pat)
        mask = mask/10
        image = self.transform(image)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = to_tensor(mask)
        return image, mask