from .base_model import DepthEstimationBase
from .modules.unet2 import SmallUNetModule

import torch
import torch.nn as nn
import torch.nn.init as init

class UNetMonocularDepthEstimator(DepthEstimationBase):
    
    def __init__(self, learning_rate, dropout_prob):
        super().__init__(learning_rate, "unet")
        self.unet = SmallUNetModule(in_channels=3, out_channels=1, dropout_prob=dropout_prob)
        self._init_unet_weights()

    def _init_unet_weights(self):
        for m in self.unet.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.unet(rgb)
