import torch
import torch.nn as nn
import lightning as L
from .modules.vit import ViT
import torch.nn.functional as F
from .base_model import DepthEstimationBase
from lightning.pytorch.utilities import grad_norm

class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.right1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.up(x)
        x1 = self.right1(x)
        return x1

class UNetViT(DepthEstimationBase):
    def __init__(self, learning_rate, img_height, img_width, patch_height, patch_width):
        super().__init__(learning_rate, "unet_vit")

        self.patch_height = patch_height
        self.patch_width = patch_width

        p_top = 0
        p_bot = 0
        if ((img_height % patch_height) != 0):
            rem = patch_height - (img_height % patch_height)
            p_top = rem//2
            p_bot = rem-p_top
        p_left = 0
        p_right = 0
        if ((img_width % patch_width) != 0):
            rem = patch_width - (img_width % patch_width)
            p_left = rem//2
            p_right = rem - p_left

        self.padding = (p_left, p_right, p_top, p_bot)
        self.unpadding = (-p_left, -p_right, -p_top, -p_bot)

        self.vit = ViT(
            image_size=(img_height + p_top + p_bot, img_width + p_left + p_right),
            patch_size=(patch_height,patch_width),
            dim=512,
            depth=1,
            heads=2,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.up2 = UpSampleLayer(512, 256)
        self.up3 = UpSampleLayer(256, 128)
        self.up4 = UpSampleLayer(128, 64)
        self.up5 = UpSampleLayer(64, 32)
        self.up6 = UpSampleLayer(32, 16)
        self.up = nn.Sequential(
            self.up2,
            self.up3,
            self.up4,
            self.up5,
            self.up6
        )
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0)
    
    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:

        x = F.pad(rgb_image, self.padding).to(torch.float32)
        print(x.shape[2],x.shape[3])
        in_height, in_width = x.shape[2], x.shape[3]

        x: torch.Tensor = self.vit(x)[:,:-1,:]
        x = x.reshape(shape=(x.shape[0], in_height//self.patch_height, in_width//self.patch_width, 512))
        x = x.permute(dims=(0,3,1,2))
        x = self.up(x)
        x = self.final_conv(x)
        x = F.pad(x, self.unpadding)
        return torch.exp(x)
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self,norm_type=2)
        #print(norms)

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.learning_rate, 
                                weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.learning_rate/10,
            T_max = 50,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 20,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }