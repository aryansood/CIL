import torch
from utils.dataset import DepthDataset
from utils.constants import DATA_DIR
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.base_model import DepthEstimationBase
from torchinfo import summary
import utils.constants as C
import albumentations as A
from pathlib import Path
from typing import List

def begin_training_loop(
    model: DepthEstimationBase,
    data_dir: Path,
    augmentations: List[A.BasicTransform],
    batch_size: int = 4,
    num_worker: int = 16,
    num_epochs: int = 5,
    random_seed: int = 80,
):
    seed_everything(random_seed, workers=True)
    torch.set_float32_matmul_precision('high')


    dataset = DepthDataset(data_dir=data_dir, augmentations=augmentations)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    summary(model, input_size=(batch_size, 3, C.H, C.W))

    wandb_logger = WandbLogger(project='monocular_depth_estimation')
    checkpoint = ModelCheckpoint(monitor="valid_silog_loss", 
                                 save_top_k=2,
                                 mode="max", 
                                 filename=model.name+"_checkpoint--{epoch}-{name}-{valid_silog_loss:.4f}")

    trainer = Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint],
        deterministic=False
    )

    trainer.fit(model, train_loader, val_loader)