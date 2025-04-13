import torch
from torchvision import transforms
import albumentations as albume
from dataset import DepthDataset
from train import train_net, test_net
from utils.constants import NUM_EPOCHS, DATA_DIR
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

seed_everything(42, workers=True)
torch.set_float32_matmul_precision('high')

device = torch.device("cuda")

wandb.init(project="my-pytorch-training")
wandb.config = {
    "epochs": 5,
    "lr": 0.001,
    "batch_size": 64
}

generator = torch.Generator().manual_seed(42)

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_array = albume.Compose([
    albume.Resize(256, 256)
])

dataset = DepthDataset(DATA_DIR, output_mask_files, transform=transform, transform_numpy=transform_array)
train_size, test_size = int(len(dataset) * 0.8), len(dataset) - (int(len(dataset) * 0.8))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
#test_net(train_dataset, device, test_size)
train_net(train_dataset, device, NUM_EPOCHS, wandb)