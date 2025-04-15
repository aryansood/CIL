import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import albumentations as albume
import cv2
from data_loader import TrainImageDataset
from loss_def import SILogLoss, GradientLossLog, SIRMSELoss
from models import ResnetUnetDecoder, ResnetTransformerUnet


def TrainResnet(dataloader, device, num_epoch, wandb = None):
    model_resnet = ResnetUnetDecoder(out_classes=1).to(device)
    model_resnet.train()
    optimizer = torch.optim.Adam(model_resnet.parameters(), lr=1e-4)
    for epoch in range(0, num_epoch):
        for batch_img, batch_mask in dataloader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            outputs = model_resnet(batch_img)
            torch.save(model_resnet.state_dict(),'weights/model_weights_12.pth')
            if wandb != None:
                pass
                #wandb.log({"loss": loss.item(), "epoch": epoch})

def train_resnet_transf_unet(dataloader, device, num_epoch, wandb = None):
    model = ResnetTransformerUnet(out_classes=1).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(0, num_epoch):
        for batch_img, batch_mask in dataloader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            outputs = model(batch_img)
            loss_fake_val = SIRMSELoss(outputs, batch_mask)
            loss = SILogLoss(outputs, batch_mask) + 0.5*GradientLossLog(outputs, batch_mask)
            loss.backward()
            optimizer.step()
            print("Epoch:", epoch)
            print("Loss_SI: ", loss.item())
            print("Loss SIRM: ", loss_fake_val)
            torch.save(model.state_dict(),'weights/model_weights.pth')
            if wandb != None:
                pass
                #wandb.log({"loss": loss.item(), "epoch": epoch})

            
            
