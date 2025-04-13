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
from dataset_create import LazyImageDataset
from loss_def import Si_Log_Loss, Loss_gradient, eval_net, loss_smrse
from model_resnet import restnet_u

def train_net(train_dataset, device, num_epoch, test_dataset, wandb):
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False)
    model_u = restnet_u(n_channels=3, n_classes=1).to(device)
    model_u.train()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_u.parameters(), lr=1e-4)
    for epoch in range(0,num_epoch):
        running_loss = 0.0
        for batch_img, batch_mask in dataloader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            outputs = model_u(batch_img)
            
            loss_train = eval_net(outputs, batch_mask)

            loss1 = Loss_gradient(torch.exp(outputs), batch_mask)
            loss = loss_smrse(outputs, batch_mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Epoch:", epoch)
            print("Loss_SI: ", loss.item())
            print("Val_Error:", loss_train)
            torch.save(model_u.state_dict(),'/home/aryan-sood/Documents/CIL/model_weights_12.pth')
            wandb.log({"loss": loss.item(), "epoch": epoch})
    
