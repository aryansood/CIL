import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ResnetUnetDecoder
import pandas as pd 
from data_loader import TestImageDataset

directory = 'ethz-cil-monocular-depth-estimation-2025'
device = torch.device("cuda")
files = os.listdir(directory)
input_images_files = [directory+ file for file in files if file.endswith('.png')]
input_images_files = sorted(input_images_files)
file_depth_ext = pd.read_csv("test_list.txt", sep=' ')['depth_paths']

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = TestImageDataset(input_images_files,transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
modelRes = ResnetUnetDecoder(n_channels=3, n_classes=1).to(device)
modelRes.load_state_dict(torch.load('weights/model_weights_12.pth', weights_only=True))

with torch.no_grad():
    for batch_idx, (batch_image) in enumerate(dataloader):
        batch_image = batch_image.to('cuda')
        outputs = modelRes(batch_image)
        for el in range(0, len(batch_image)):
            numpy_arr = outputs[el][0].detach().cpu().numpy()
            np.save(osp.join(directory, file_depth_ext[batch_idx*16 + el]), numpy_arr)
    
