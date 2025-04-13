import torch
from torchvision import transforms
from dataset import DepthDataset
from utils.constants import DATA_DIR
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchinfo import summary
import utils.constants as c

from models import UNetMonocularDepthEstimator

NUM_EPOCHS = 2
BATCH_SIZE = 2
seed_everything(80, workers=True)
torch.set_float32_matmul_precision('high')


input_transform = transforms.Compose([
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
])


dataset = DepthDataset(DATA_DIR, transform=input_transform, target_transform=target_transform)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)


model = UNetMonocularDepthEstimator(learning_rate=0.0005, dropout_prob=0.3)
summary(model, input_size=(BATCH_SIZE, 3, c.H, c.W))

wandb_logger = WandbLogger(project='monocular_depth_estimation')
checkpoint = ModelCheckpoint(monitor="valid_silog_loss", save_top_k=2, mode="max", filename="checkpoint-{epoch}-{valid_silog_loss:.4f}")

trainer = Trainer(
    max_epochs=20,
    logger=wandb_logger,
    callbacks=[checkpoint],
    deterministic=False
)

trainer.fit(model, train_loader, val_loader)