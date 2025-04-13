import numpy as np
import os
import os.path as osp
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
from model_resnet import restnet_u
import pandas as pd 

class TestImageDataset(Dataset):
    def __init__(self, image_paths,transform=None, transform_numpy = None):
        self.image_paths = image_paths
        self.transform = transform
        self.transform_numpy = transform_numpy

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_path
    
directory = '/home/aryan-sood/Documents/CIL/ethz-cil-monocular-depth-estimation-2025/test/test/'
device = torch.device("cuda")
files = os.listdir(directory)
input_images_files = [directory+ file for file in files if file.endswith('.png')]
input_images_files = sorted(input_images_files)
file_depth_ext = pd.read_csv("ethz-cil-monocular-depth-estimation-2025/test_list.txt", sep=' ')['depth_paths']

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = TestImageDataset(input_images_files,transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
model_u = restnet_u(n_channels=3, n_classes=1).to(device)
model_u.load_state_dict(torch.load('/home/aryan-sood/Documents/CIL/model_weights_12.pth', weights_only=True))
with torch.no_grad():
    for batch_idx, (batch_image, batch_path) in enumerate(dataloader):
        batch_image = batch_image.to('cuda')
        outputs = model_u(batch_image)
        #outputs = torch.exp(outputs)
        for el in range(0, len(batch_image)):
            numpy_arr = outputs[el][0].detach().cpu().numpy()
            #plt.imshow(numpy_arr)
            #plt.show()
            np.save(osp.join(directory, file_depth_ext[batch_idx*16 + el]),numpy_arr)
    
