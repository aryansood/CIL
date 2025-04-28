import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
from torchvision import transforms
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    RBG_COL = 'rgb_paths'
    DPT_COL = 'depth_paths'

    # This selects wheter the dataset will contain the Train or Test images. It indicates the relative paths from DATA_DIR to 
    # (1) the folder containing the rgb_images and depth_masks (if applicable)  
    # (2) the list of pairs of images and mask paths
    # (3) whether the folder contains the depth_masks
    TEST =  'test/test',  'test_list.txt', False
    TRAIN = 'train/train', 'train_list.txt', True

    def __init__(self, data_dir, mode, transform=None, target_transform=None):
        if (mode != DepthDataset.TEST) and (mode != DepthDataset.TRAIN): raise TypeError("DepthDataset mode must be either DepthDataset.TRAIN or DepthDataset.TEST!")

        self.mask_dir = Path(data_dir) / mode[0]
        self.mask_list_path = Path(data_dir) / mode[1]
        self.has_gt = mode[2]
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_paths = pd.read_csv(self.mask_list_path, sep=' ', names=[DepthDataset.RBG_COL, DepthDataset.DPT_COL])
        
    def get_path(self, idx, col):
        return self.mask_dir / self.data_paths.at[idx, col]

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx:int) -> Union[Tuple[Image.Image, torch.Tensor, str], Tuple[Image.Image, str]]:
        
        rgb = Image.open(self.get_path(idx, DepthDataset.RBG_COL)).convert('RGB')
        
        if self.has_gt:
            depth = np.load(self.get_path(idx, DepthDataset.DPT_COL), allow_pickle=True).astype(np.float32)
            depth = torch.from_numpy(depth)
            
            if self.transform:        rgb = self.transform(rgb)
            if self.target_transform: depth = self.target_transform(depth)
            
            # rgb_image, ground truth and image_path (might be needed for saving output +samples)
            return rgb, depth
        
        else:
            return rgb#, self.get_path(idx, DepthDataset.RBG_COL)

    def randomize(self):
        self.data_paths = self.data_paths.sample(frac=1).reset_index(drop=True)



# Wrapper for Randomized DepthDataset
class RandomizedDataset:
    def __init__(self, data_dir, mode):
        self.dataset = DepthDataset(data_dir, mode)
        self.mode = mode
        self.idx = -1

        self.dataset.randomize()

    def random_entry(self):
        self.idx += 1
        return self.dataset[self.idx]

    def random_image(self):
        self.idx += 1
        return self.dataset[self.idx][0]

    def random_mask(self):
        if (self.mode == DepthDataset.TEST): raise ValueError("Impossible to access ground truth of test image")
        self.idx += 1
        return self.dataset[self.idx][1]
    
    def iter(self, iterations):
        if (self.mode == DepthDataset.TEST):
            for i in range(iterations):
                yield self.random_entry()[0]
        else:
            for i in range(iterations):
                yield self.random_entry()[0:2]




if __name__ == '__main__':
    # Example Usage, worked up to changes on 14.04.2025
    from torch.utils.data import random_split
    from utils.constants import DATA_DIR

    dataset = DepthDataset(DATA_DIR, DepthDataset.TRAIN, transform=None, target_transform=None)
    print(dataset[0])
    
    train_dataset, val_dataset = random_split(dataset, [0.8,0.2], torch.Generator().manual_seed(42))
    print(f"\nLegths: {len(train_dataset)} + {len(val_dataset)} = {len(dataset)}\n")
    
    dataset.randomize()
    print(dataset.data_paths.at[0, DepthDataset.RBG_COL], dataset.data_paths.at[0, DepthDataset.DPT_COL])
    print(dataset.data_paths.at[1, DepthDataset.RBG_COL], dataset.data_paths.at[1, DepthDataset.DPT_COL])
    print(dataset.data_paths.at[2, DepthDataset.RBG_COL], dataset.data_paths.at[2, DepthDataset.DPT_COL])

    test_dataset = DepthDataset(DATA_DIR, DepthDataset.TEST, transform=None, target_transform=None)
    print("\n", test_dataset[0])