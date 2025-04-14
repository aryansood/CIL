import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TrainImageDataset(Dataset):
    """
    Dataset creation for training.
    """
    def __init__(self, image_paths, mask_path,transform=None, transform_mask = None):
        self.image_paths = image_paths
        self.mask_path = mask_path
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        mask_path = self.mask_path[idx]
        mask = np.load(mask_path)
        mask = self.transform_mask(mask)

        return image, mask

class TestImageDataset(Dataset):
    """
    Dataset creation for Test.
    """
    def __init__(self, image_paths,transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image