from .base_model import DepthEstimationBase
from transformers import Mask2FormerModel
import torch.nn as nn
import torch

class Mask2FormerDepthEstimator(DepthEstimationBase):

    def __init__(self, learning_rate, 
                 pretrained_weights = "facebook/mask2former-swin-small-coco-instance",
                 use_silog = True):
        super().__init__(learning_rate, "maskformer_depth")
        self.maskformer_model = Mask2FormerModel.from_pretrained(pretrained_weights)
        self.mask_fusion = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        )
        self.use_silog = use_silog
        self.return_log = False
        self.optimizer = torch.optim.AdamW(
            weight_decay=1e-3,
            lr = self.learning_rate,
            params=self.parameters()
        )
        self.optimizer_config = {
            "optimizer": self.optimizer
        }


    def forward(self, x, output_mask_logits = False):
        mask_logits = self.maskformer_model(x)["masks_queries_logits"]
        up_mask_logits = nn.functional.interpolate(
            mask_logits[-1], size=(426,560), mode="bilinear", align_corners=False
        )

        depth_logits = self.mask_fusion(up_mask_logits).squeeze(1)

        result = depth_logits
        if not self.return_log:
            result = result.exp()
        if (output_mask_logits):
            return result, mask_logits
        return result
    
    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        
        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)

        return silog if self.use_silog else sirme
    
    def configure_optimizers(self):
        return self.optimizer_config
    