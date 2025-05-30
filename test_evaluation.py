import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.dataset import DepthDataset
from torchvision import transforms
import numpy as np
import albumentations as A
import wandb
from torch.utils.data import DataLoader

from training import begin_training_loop
from models.resnet_transformer_unet import ResnetTransformerUnet
from models.segformer_depth import SegFormerDepthEstimator
import albumentations as A
from pathlib import Path
from datetime import timedelta
import pandas as pd
import os.path as osp
import os

normalize_noise_augmentation = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentations = [
    A.Compose([
        normalize_noise_augmentation,
        A.ToTensorV2()
    ])
]

# Replace the checkpoint file (.ckpt) with the path of the checkpoint file generated during training.
# The same checkpoint file we used to achieve our best Kaggle score can be found at:
# https://drive.google.com/file/d/1iIvGZ2I5k3vbXvfXvBV6dONWtoDYHTlK/view
model = SegFormerDepthEstimator.load_from_checkpoint('segformer_depth_checkpoint.ckpt').to('cuda')
model.eval()

# `data_dir`: replace the parameter with the path containing the test data.
dataset = DepthDataset(data_dir='/cluster/courses/cil/monocular_depth/data/test', augmentations=augmentations, has_gt=False)
test_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# `directory`: replace the variable with the path where you want to save the prediction files.
directory = 'ethz-cil-monocular-depth-estimation-2025/test'
if not os.path.exists(directory):
    os.makedirs(directory)
device = torch.device("cuda")

file_depth_ext = pd.read_csv("test_list.txt", sep=' ')['depth_paths']

with torch.no_grad():
    for batch_idx, (batch_image, batch_image_path) in enumerate(test_loader):
        print(batch_image.shape)
        batch_image = batch_image.to('cuda')
        outputs = model(batch_image)
        for el in range(0, len(batch_image)):
            numpy_arr = outputs[el][0].detach().cpu().numpy()
            np.save(osp.join(directory, file_depth_ext[batch_idx*16 + el]), numpy_arr)
