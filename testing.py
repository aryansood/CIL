from models.base_model import DepthEstimationBase
from lightning import Trainer
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataset import DepthDataset
import albumentations as A
from typing import Type
import pandas as pd
import torch
import numpy as np
import os.path as osp

def create_test_prediction(
        model: Type[DepthEstimationBase],
        data_dir: Path,
        test_list: Path,
        checkpoint_path: Path,
        batch_size: int = 4,
        num_workers: int = 4,
        ):

    testing_transforms = [A.Compose(
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )]
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    test_dataset = DepthDataset(data_dir, has_gt=False, augmentations=testing_transforms)

    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    file_depth_ext = pd.read_csv(test_list, sep=' ')['depth_paths']
    with torch.no_grad():
        for batch_idx, (batch_image, batch_image_path) in enumerate(test_dl):
            batch_image = batch_image.float()
            # print(batch_image.shape)
            batch_image = batch_image.permute(0, 3, 1, 2)
            # print(batch_image.shape)
            batch_image = batch_image
            outputs = model(batch_image)
            for el in range(0, len(batch_image)):
                numpy_arr = outputs[el][0].detach().cpu().numpy()
                #plt.imshow(numpy_arr)
                #plt.show()
                np.save(osp.join(data_dir, file_depth_ext[batch_idx*batch_size + el]), numpy_arr)

