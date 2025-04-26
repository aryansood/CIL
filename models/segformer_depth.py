from .base_model import DepthEstimationBase
from transformers import SegformerModel, SegformerDecodeHead, SegformerConfig
import torch.nn as nn
import torch

class SegFormerDepthEstimator(DepthEstimationBase):

    def __init__(self, learning_rate, 
                 pretrained_weights = "nvidia/segformer-b5-finetuned-ade-640-640",
                 use_silog = True,
                 use_final_conv = False):
        super().__init__(learning_rate, "segformer_depth")
        self.encoder = SegformerModel.from_pretrained(pretrained_weights)
        self.config = self.encoder.config
        self.config.num_labels=1
        self.decoder = SegformerDecodeHead(self.config)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        )
        self.use_silog = use_silog
        self.use_final_conv = use_final_conv
        self.optimizer = torch.optim.AdamW(
            weight_decay=1e-3,
            lr = self.learning_rate,
            params=self.parameters()
        )
        self.optimizer_config = {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0 = 25,
                    eta_min=self.learning_rate/10,  
                ),
                "interval": "step",
                "frequency": 5
            }
        }


    def forward(self, x):
        encoded = self.encoder(x, output_hidden_states = True, output_attentions= False)
        logits = self.decoder(encoded.hidden_states)
        upsampled_logits = nn.functional.interpolate(
            logits, size=(426,560), mode="bilinear", align_corners=False
        )
        if self.use_final_conv:
            upsampled_logits = self.final_conv(torch.concat((upsampled_logits, x), dim=1))
        upsampled_logits = upsampled_logits.squeeze(1)
        return upsampled_logits.exp()
    
    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        
        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)
        self.log('train_lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True)

        return silog if self.use_silog else sirme
    
    def configure_optimizers(self):
        return self.optimizer_config