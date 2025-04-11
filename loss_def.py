import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def Si_Log_Loss(output_log, target):
    diff_log = output_log-target
    num_pixel = diff_log[0].numel()
    term1 = torch.square(diff_log)
    term1 = torch.mean(term1)
    term2 = torch.sum(diff_log, dim=(2,3))/num_pixel
    term2 = torch.square(term2)
    term2 = torch.mean(term2)
    
    Loss_average = term1 - 0.5*term2
    return Loss_average

def Loss_gradient(output, target):
    dy_targ, dx_targ = torch.gradient(target, dim=(2,3))
    dy_out, dx_out = torch.gradient(output, dim=(2,3))
    
    dy_targ = dy_targ[:, 0, :, :]
    dx_targ = dx_targ[:, 0, :, :]
    dy_out = dy_out[:, 0, :, :]
    dx_out = dx_out[:, 0, :, :]

    new_image_tensor = (dx_targ)**2+(dy_targ)**2
    new_image_tensor = torch.sqrt(new_image_tensor)
    output_sim = (dx_out)**2+(dy_out)**2
    output_sim = torch.sqrt(output_sim)

    canny_edge_detector_target = torch.where(new_image_tensor < 0.020, torch.tensor(0.0), new_image_tensor)

    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(canny_edge_detector_target[0].detach().cpu().numpy())
    # ax2.imshow(target[0][0].detach().cpu().numpy())
    # plt.show()

    # diff_y = (dy_targ-dy_out)**2
    # diff_x = (dx_targ-dx_out)**2

    # diff_x_mean = torch.mean(diff_x)
    # diff_y_mean = torch.mean(diff_y)
    diff_edge = torch.square(output_sim-canny_edge_detector_target)
    diff_edge = torch.sqrt(diff_edge)
    return torch.mean(diff_edge)

def eval_net(output, target):
    pixel_out = output
    #assert(pixel_out.shape == (16, 1, 256, 256) and target.shape == (16, 1, 256, 256))
    pixel_out = pixel_out.squeeze()
    target = target.squeeze()
    pixel_targ = torch.log(target)
    diff_pixel = pixel_out-pixel_targ
    alpha = torch.mean(diff_pixel, dim=[1,2], keepdim=True)
    result = diff_pixel+alpha
    result = torch.square(result)
    #assert(result.shape == (16, 256, 256))
    mean = torch.sqrt(torch.mean(result, dim=[1,2]))
    print("Shape: ", mean.shape)
    print("mean value:", mean)
    mean = torch.mean(mean)
    
    return(mean)







