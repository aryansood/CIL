import torch
from torch import nn
from torchmetrics.classification import BinaryJaccardIndex
import torch.nn.init as init
import lightning as L
from torchvision import transforms

from utils.losses import SILogLoss, SIRMSELoss
from modules.unet import UNetModule
from modules.unet2 import SmallUNetModule

class UNetMonocularDepthEstimator(L.LightningModule):
    def __init__(self, learning_rate: float = 0.00005, dropout_prob=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.unet = SmallUNetModule(in_channels=3, out_channels=1, dropout_prob=dropout_prob)
        self.criterionSILog = SILogLoss
        self.criterionSIRME = SIRMSELoss

        self._init_unet_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
    
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

    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)

        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)

        return sirme

    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch

        Y_prob = self(X)

        # assert torch.all((Y_prob >= 0) & (Y_prob <= 1)), "Input values should be in the range [0, 1]"
        # assert torch.all((Y == 0) | (Y == 1)), "Target values should be 0 or 1"
        # assert Y_prob.shape == Y.shape, "Input and target must have the same shape"

        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('valid_silog_loss', silog, prog_bar=True)
        self.log('valid_sirme_loss', sirme, prog_bar=True)

        return sirme

    def test_step(self, batch, batch_idx):
        X, Y = batch

        Y_prob = self(X)

        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('test_silog_loss', silog, prog_bar=True)
        self.log('test_sirme_loss', sirme, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        return optimizer

if __name__ == "__main__":
    UNetMonocularDepthEstimator()