import torch
import torch.nn as nn
import lightning as L
from .modules.vit import ViT
import torch.nn.functional as F
from .base_model import DepthEstimationBase
from lightning.pytorch.utilities import grad_norm
from torchvision.models import swin_b, Swin_B_Weights


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.right1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.up(x)
        x1 = self.right1(x)
        return x1
    
class UNetSwin(DepthEstimationBase):
    def __init__(self, learning_rate):
        super().__init__(learning_rate, "unet_vit")


        swin_full = swin_b(Swin_B_Weights.IMAGENET1K_V1)
        self.swin_model = nn.Sequential(
            swin_full.features,
            swin_full.norm,
            swin_full.permute
        )

        self.up = nn.Sequential(
            UpSampleLayer(1024, 512),
            UpSampleLayer(512, 128),
            UpSampleLayer(128, 64),
            UpSampleLayer(64, 32),
            UpSampleLayer(32, 16),
        )
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0)
    
    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        
        img = rgb_image.to(torch.float32)

        with torch.no_grad():
            img: torch.Tensor = self.swin_model(img)
        x = self.up(img)
        x = self.final_conv(x)
        delta_h = rgb_image.shape[-2] - x.shape[-2] 
        delta_w = rgb_image.shape[-1] - x.shape[-1]
        x = F.pad(x, (delta_w//2, delta_w - delta_w//2, delta_h//2, delta_h-delta_h//2))
        return x
    

    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        loss = nn.L1Loss(reduction="mean")(Y_prob.squeeze(1),Y.log())
        self.log("train mae", loss, prog_bar=True)
        with torch.no_grad():
            silog = self.criterionSILog(Y_prob.exp(), Y)
            sirme = self.criterionSIRME(Y_prob.exp(), Y)
            self.log('train_silog_loss', silog, prog_bar=True)
            self.log('train_sirme_loss', sirme, prog_bar=True)
        return loss

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