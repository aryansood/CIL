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
from loss_def import Si_Log_Loss, Loss_gradient, eval_net, loss_smrse, eval_net_test, SIRMSELoss
from model_resnet import restnet_u


def test_net(test_dataset, device, test_size):
    model_u = restnet_u(n_channels=3, n_classes=1).to(device)
    model_u.load_state_dict(torch.load('/home/aryan-sood/Documents/CIL/model_weights_12.pth', weights_only=True))
    dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
    error_arr = torch.empty(0)
    error_arr = error_arr.to('cuda')
    tot = 0
    for batch_idx, (batch_img, batch_mask) in enumerate(dataloader_test):
       
        batch_img = batch_img.to(device)
        with torch.no_grad():
            outputs = model_u(batch_img)
            batch_mask =batch_mask.to('cuda')
            #error = eval_net_test(outputs, batch_mask)
            error = SIRMSELoss(torch.exp(outputs), batch_mask)
            error_arr = torch.cat((error_arr, error))
            tot += torch.mean(error)
            print(tot/(batch_idx+1))
    mean_var = torch.var_mean(error_arr)
    print("Vra and mean: ", mean_var)
    print("Shape", error_arr.shape)
    

    #         print(error)
    #         avrg = avrg + error
    #         print(avrg)
    # print("Final: ", avrg/test_size)