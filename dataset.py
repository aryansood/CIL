import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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


class DepthDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, has_gt=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        
        data_paths = os.listdir(data_dir)
        self.rgb_paths   = sorted([os.path.join(data_dir, file) for file in data_paths if file.endswith('.png')])
        self.depth_paths = sorted([os.path.join(data_dir, file) for file in data_paths if file.endswith('.npy')])

    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        
        if self.has_gt:
            depth = np.load(self.depth_paths[idx]).astype(np.float32)
            depth = torch.from_numpy(depth)
            
            if self.transform:        rgb = self.transform(rgb)
            if self.target_transform: depth = self.target_transform(depth)
            else:                     depth = depth.unsqueeze(0)
            
            return rgb, depth, self.file_pairs[idx][0]
        
        else:
            return rgb, self.file_list[idx]  

    def randomize(self):
        random.shuffle(self.file_pairs if self.has_gt else self.file_list)


if __name__ == '__main__':
    # Example Usage
    from torch.utils.data import random_split
    from constants import DATA_DIR

    train_dataset = DepthDataset(DATA_DIR, transform=None, target_transform=None, has_gt=True)
    train_dataset, test_dataset = random_split(train_dataset, [8,2], torch.Generator().manual_seed(42))


    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(range(10), [8, 2], generator=generator1)
    print(list(train_dataset), list(test_dataset))