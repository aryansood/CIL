import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
            
            # rgb_image, ground truth and image_path (might be needed for saving output +samples)
            return rgb, depth, self.rgb_paths[idx][0] #
        
        else:
            return rgb, self.rgb_paths[idx]  

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
    from constants import DATA_DIR

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
