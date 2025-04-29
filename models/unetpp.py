import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .base_model import DepthEstimationBase
from .modules.unetpp import UNetPlusPlusModule


class UNetPlusPlus(DepthEstimationBase):
    def __init__(self, learning_rate, dropout_prob):
        super().__init__(learning_rate=learning_rate, name="UNet++")
        self.unet = UNetPlusPlusModule(in_channels=3, out_channels=1, dropout_prob=dropout_prob)
        self._init_unet_weights()
        
    def _init_unet_weights(self):
        for m in self.unet.modules():
            if isinstance(m, nn.Conv2d):
                init.dirac_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        
        silog = self.criterionSILog(Y_prob, Y, is_output_logarithm=True)
        sirme = self.criterionSIRME(Y_prob, Y, is_output_logarithm=True)
        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)

        return sirme
    
    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)

        # assert torch.all((Y_prob >= 0) & (Y_prob <= 1)), "Input values should be in the range [0, 1]"
        # assert torch.all((Y == 0) | (Y == 1)), "Target values should be 0 or 1"
        # assert Y_prob.shape == Y.shape, "Input and target must have the same shape"
        silog = self.criterionSILog(Y_prob, Y, is_output_logarithm=True)
        sirme = self.criterionSIRME(Y_prob, Y, is_output_logarithm=True)
        self.log('valid_silog_loss', silog, prog_bar=True)
        self.log('valid_sirme_loss', sirme, prog_bar=True)

        return sirme

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        result = self.unet.forward(rgb, return_log=True)
        return result
