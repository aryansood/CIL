import torch
import torch.nn as nn
import lightning as L
from .modules.vit import ViT
import torch.nn.functional as F
from .base_model import DepthEstimationBase
from lightning.pytorch.utilities import grad_norm
import torchvision
from .modules.models_parts import *


class ResnetUnet(DepthEstimationBase):
    """
    Model doing the following: Resnet+transformer+decoder.
    """
    def __init__(self, learning_rate):
        super().__init__(learning_rate=learning_rate, name="resnet_transformer_unet")
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base_layers = list(self.resnet50.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        self.up1 = (DecoderUnetSkip(3072, 1024))
        self.up2 = (DecoderUnetSkip(1536, 512))
        self.up3 = (DecoderUnetSkip(768, 256))
        self.up4 = (DecoderUnetSkip(320, 128))
        self.up5 = (DecoderUnet(128, 64))
        self.outc = (OutConv(64, 1))
        
    def forward(self, x):
    #with torch.no_grad():
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up5(x)
        logits = self.outc(x)
        logits = torch.exp(logits)
        return logits