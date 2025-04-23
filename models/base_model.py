import pytorch_lightning as L
from pytorch_lightning.utilities import grad_norm
import torch

from utils.losses import SILogLoss, SIRMSELoss

class DepthEstimationBase(L.LightningModule):

    def __init__(self, learning_rate, name):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.criterionSILog = SILogLoss
        self.criterionSIRME = SIRMSELoss
        self.name = name

    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("")

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

        return sirme
    
    def on_before_optimizer_step(self, _):
        norms = grad_norm(self,norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate, 
                                     )
        return optimizer
