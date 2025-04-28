import torch
from utils.dataset import DepthDataset
from utils.constants import DATA_DIR
from pytorch_lightning import Trainer, seed_everything  
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.base_model import DepthEstimationBase
from torchinfo import summary
import utils.constants as C
import albumentations as A
from pathlib import Path
from typing import List
from datetime import timedelta
import math

def begin_training_loop(
    model: DepthEstimationBase,
    data_dir: Path,
    augmentations: List[A.BasicTransform],
    random_split: bool = True,
    train_split = None,
    val_split = None,
    batch_size: int = 2,
    num_worker: int = 4,
    num_epochs: int = 5,
    random_seed: int = 80,
    gradient_clip_val: int = 0,
    gradient_clip_algorithm: str = "norm",
    check_point_every_step: int = 500,
    debugging: bool = False,
    max_training_duration: timedelta = timedelta(hours=4),
    effective_batch_size: int = 0):
    seed_everything(random_seed, workers=True)
    torch.set_float32_matmul_precision('high')


    dataset = DepthDataset(data_dir=data_dir, augmentations=augmentations)
    if (random_split):
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    else:
        train_dataset = DepthDataset(data_dir=data_dir, data_paths=train_split, augmentations=augmentations)
        val_dataset = DepthDataset(data_dir=data_dir, data_paths=val_split, augmentations=augmentations)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    summary(model, input_size=(batch_size, 3, C.H, C.W))

    wandb_logger = WandbLogger(project='monocular_depth_estimation')

    #callbacks
    validation_checkpoint = ModelCheckpoint(monitor="valid_sirme_loss", 
                                 save_top_k=2,
                                 mode="min", 
                                 filename=model.name+"_checkpoint--{epoch}-{valid_sirme_loss:.4f}")
    latest_checkpoint = ModelCheckpoint(
        monitor="step",
        mode="max",
        every_n_train_steps=check_point_every_step,
        save_top_k=2,
        filename=model.name+"_checkpoint-{epoch}-{step}-{train_silog_loss:.4f}"
    )
    timer = Timer(
        duration=max_training_duration
    )

    if effective_batch_size < batch_size:
        effective_batch_size = batch_size

    if debugging:
        trainer = Trainer(
            max_epochs=num_epochs,
            logger=wandb_logger,
            callbacks=[timer],
            deterministic=False,
            overfit_batches=2,
            detect_anomaly=True,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
    else:
        trainer = Trainer(
            max_epochs=num_epochs,
            logger=wandb_logger,
            callbacks=[validation_checkpoint, 
                    latest_checkpoint,
                    timer],
            deterministic=False,
            accumulate_grad_batches=math.ceil(effective_batch_size/batch_size),
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    trainer.fit(model, train_loader, val_loader)