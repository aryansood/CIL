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
from loss_def import Si_Log_Loss, Loss_gradient, eval_net
from model_resnet import restnet_u



def train_net(train_dataset, device, num_epoch, wandb):
    
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False)
    model_u = restnet_u(n_channels=3, n_classes=1).to(device)
    model_u.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_u.parameters(), lr=1e-4)
    for epoch in range(0,num_epoch):
        running_loss = 0.0
        for batch_img, batch_mask in dataloader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)
            #print(torch.max(batch_mask[0][0]))
            optimizer.zero_grad()
            outputs = model_u(batch_img)
            #assert(outputs.shape[0] == 16)
            #loss1 = Loss_gradient(torch.exp(outputs), batch_mask)
            loss = Si_Log_Loss(torch.exp(outputs), batch_mask)#+loss1#criterion(torch.exp(outputs), batch_mask)+loss1#Si_Log_Loss(outputs, batch_mask)#criterion(outputs, batch_mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Epoch:", epoch)
            print("Loss_SI_LOG: ", loss.item())
            loss_train = eval_net(outputs, batch_mask)
            print("criterion:", loss_train)
            torch.save(model_u.state_dict(),'/home/aryan-sood/Documents/CIL/model_weights_8.pth')
            wandb.log({"loss": loss.item(), "epoch": epoch})
    
def test_net(test_dataset, device, test_size):
    model_u = restnet_u(n_channels=3, n_classes=1).to(device)
    model_u.load_state_dict(torch.load('/home/aryan-sood/Documents/CIL/model_weights_8.pth', weights_only=True))
    
    pp = 0
    for elem in list(model_u.parameters()):
        nn = 1
        for s in list(elem.size()):
            nn = nn*s
        pp+=nn
    print(pp)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    for batch_img, batch_mask in dataloader_test:
        batch_img = batch_img.to(device)
        avrg = 0
        with torch.no_grad():
            outputs = model_u(batch_img)
            batch_mask =batch_mask.to('cuda')
            error = eval_net(outputs, batch_mask)
            print(error)
            avrg += error
    print(avrg/test_size)
        #     Loss_gradient(outputs, batch_img)
        # mask_pred = outputs[0][0].detach().cpu().numpy()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.imshow(mask_pred)
        # ax2.imshow(batch_mask[0][0].detach().cpu().numpy())
        # plt.show()