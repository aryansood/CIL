from training import begin_training_loop
from models.segformer_depth import SegFormerDepthEstimator
import albumentations as A
from pathlib import Path
from datetime import timedelta
import torch
import pandas as pd
import numpy as np
import gc
import os 
import os.path as osp
import wandb
#Insert wandb_key if wanted to use.
wandb_key = ''

wandb.login(
    key=wandb_key
)
#Insert Path of the location of the data before train/train
DS_PATH = ""

print(os.listdir(DS_PATH))

TRAIN_PATH = osp.join(DS_PATH, "train/train")

normalize_noise_augmentation = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentations = [
    A.Compose([
        normalize_noise_augmentation,
        A.ToTensorV2()
    ])
]

model = SegFormerDepthEstimator(1e-4, 
                                pretrained_weights="nvidia/segformer-b5-finetuned-ade-640-640", 
                                use_final_conv=False)

model.optimizer_config = {
        "optimizer": model.optimizer,
        "lr_scheduler": {
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            model.optimizer,
            T_0 = 100,
            #eta_min=model.learning_rate/10,  
        ),
        "interval": "step",
        "frequency": 50
    }
}

begin_training_loop(
    model = model,
    data_dir = Path(TRAIN_PATH),
    random_split=False,
    train_split=pd.read_csv("train_split.csv")["file_name"].to_list(),
    val_split=pd.read_csv("val_split.csv")["file_name"].to_list(),
    augmentations = augmentations,
    batch_size=2,
    num_worker=2,
    max_training_duration = timedelta(hours=10),
    check_point_every_step = 3000,
    effective_batch_size = 16
)

wandb.finish()