import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset
import pandas as pd
from typing import List
import albumentations as A
from pathlib import Path

class DepthDataset(Dataset):
    def __init__(self, 
                 data_dir,
                 data_paths = None, 
                 has_gt=True,
                 augmentations: List[A.BasicTransform] = [A.Compose([])]):
        
        self.data_dir = data_dir
        self.has_gt = has_gt
        self.augmentations = augmentations
        
        if (data_paths is None):
            data_paths = os.listdir(data_dir)
        self.augment = True
        if (augmentations is None):
            self.augment = False

        self.rgb_paths = sorted([os.path.join(data_dir, file) for file in data_paths if file.endswith('.png')])
        if self.has_gt:
            self.depth_paths = sorted([os.path.join(data_dir, file) for file in data_paths if file.endswith('.npy')])
            self.df: pd.DataFrame = pd.DataFrame().from_dict(
                {"rgb": self.rgb_paths, "depth": self.depth_paths}
            )
            if (self.augment):
                aug_len = len(self.augmentations) if self.augmentations is not None else 0
                aug_columns = pd.DataFrame().from_dict({
                        "aug_id": [i//len(self.df) for i in range(0, len(self.df)*(aug_len))]
                    }
                )
                replicated = pd.DataFrame(np.repeat(self.df.values, aug_len, axis=0), columns=["rgb", "depth"])
                self.df = pd.concat([replicated, aug_columns], axis=1)
        else:
            self.df: pd.DataFrame = pd.DataFrame().from_dict(
                {"rgb": self.rgb_paths}
            )
            if (self.augment):
                aug_len = len(self.augmentations) if self.augmentations is not None else 0
                aug_columns = pd.DataFrame().from_dict({
                        "aug_id": [i//len(self.df) for i in range(0, len(self.df)*(aug_len))]
                    }
                )
                replicated = pd.DataFrame(np.repeat(self.df.values, aug_len, axis=0), columns=["rgb"])
                self.df = pd.concat([replicated, aug_columns], axis=1) 


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        rgb_path = self.df.iloc[idx]["rgb"]
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        
        if self.has_gt:
            depth_path = self.df.iloc[idx]["depth"]
            depth = np.load(depth_path).astype(np.float32)
            #depth = torch.from_numpy(depth)
            if self.augment:
                aug_idx = self.df.iloc[idx]["aug_id"]
                augmentation = self.augmentations[aug_idx]
                augmented = augmentation(image=rgb, mask=depth)
                rgb, depth = augmented["image"], augmented["mask"]

            # rgb_image, ground truth and image_path (might be needed for saving output +samples)
            return rgb, depth, rgb_path 
        
        else:
            if self.augment:
                aug_idx = self.df.iloc[idx]["aug_id"]
                augmentation = self.augmentations[aug_idx]
                augmented = augmentation(image=rgb)
                rgb = augmented["image"]
            return rgb, rgb_path
    
    def randomize(self):
        if self.has_gt:
            joint_pairs = list(zip(self.rgb_paths, self.depth_paths))
            random.shuffle(joint_pairs)
            self.rgb_paths, self.depth_paths = zip(*joint_pairs)
        else:
            random.shuffle(self.rgb_paths)



if __name__ == '__main__':
    # Example Usage
    from torch.utils.data import random_split
    from .constants import DATA_DIR

    dataset = DepthDataset(DATA_DIR, transform=None, target_transform=None, has_gt=True)
    train_dataset, test_dataset = random_split(dataset, [0.8,0.2], torch.Generator().manual_seed(42))
    
    print(f"Legths: {len(train_dataset)} + {len(test_dataset)} = {len(dataset)}")
    
    dataset.randomize()
    print(dataset.rgb_paths[0], dataset.depth_paths[0])
    print(dataset.rgb_paths[1], dataset.depth_paths[1])
    print(dataset.rgb_paths[2], dataset.depth_paths[2])




# class LazyImageDataset(Dataset):
#     def __init__(self, image_paths, mask_path, transform=None, transform_numpy = None):
#         self.image_paths = image_paths
#         self.mask_path = mask_path
#         self.transform = transform
#         self.transform_numpy = transform_numpy

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         mask_pat = self.mask_path[idx]
#         to_tensor = transforms.ToTensor()
#         image = Image.open(img_path).convert("RGB")
#         mask = np.load(mask_pat)
#         mask = mask/10
#         image = self.transform(image)
#         # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
#         mask = to_tensor(mask)
#         return image, mask
