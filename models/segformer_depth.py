from .base_model import DepthEstimationBase
from transformers import SegformerModel, SegformerDecodeHead, SegformerConfig
import torch.nn as nn

class SegFormerDepthEstimator(DepthEstimationBase):

    def __init__(self, learning_rate, 
                 pretrained_weights = "nvidia/segformer-b5-finetuned-ade-640-640",
                 use_silog = True):
        super().__init__(learning_rate, "segformer_depth")
        self.encoder = SegformerModel.from_pretrained(pretrained_weights)
        self.config = self.encoder.config
        self.config.num_labels=1
        self.decoder = SegformerDecodeHead(self.config)
        self.use_silog = use_silog

    def forward(self, x):
        encoded = self.encoder(x, output_hidden_states = True, output_attentions= True)
        logits = self.decoder(encoded.hidden_states)
        upsampled_logits = nn.functional.interpolate(
            logits, size=(426,560), mode="bilinear", align_corners=False
        ).squeeze(1)
        #print(upsampled_logits.shape)
        return upsampled_logits.exp()
    
    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        
        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)
        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)

        return silog if self.use_silog else sirme