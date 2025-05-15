from .base_model import DepthEstimationBase
from transformers import SegformerModel, SegformerDecodeHead
import torch.nn as nn
import torch
from .modules.svd_linear import LowRankLinear, LowRankConv1x1

class LowRankSegFormer(DepthEstimationBase):

    def __init__(self, learning_rate, 
                 pretrained_weights = "nvidia/segformer-b5-finetuned-ade-640-640",
                 use_silog = True,
                 lambda_orth = 1.0,
                 lambda_hoyer = 0.5):
        super().__init__(learning_rate, "segformer_depth")
        self.encoder = SegformerModel.from_pretrained(pretrained_weights)
        self.config = self.encoder.config
        self.config.num_labels=1
        self.decoder = SegformerDecodeHead(self.config)
        
        for layer in self.decoder.linear_c:
            layer.proj = LowRankLinear(input_dim=layer.proj.in_features, 
                                   output_dim=layer.proj.out_features, 
                                   layer=layer.proj)
        self.decoder.linear_fuse = LowRankConv1x1(in_channels=-1,out_channels=-1,layer=self.decoder.linear_fuse)

        self.use_silog = use_silog
        self.optimizer = torch.optim.AdamW(
            weight_decay=1e-3,
            lr = self.learning_rate,
            params=self.parameters()
        )

        self.lambda_orth = lambda_orth
        self.lambda_hoyer = lambda_hoyer


    def forward(self, x):
        encoded = self.encoder(x, output_hidden_states = True, output_attentions= False)
        logits = self.decoder(encoded.hidden_states)
        upsampled_logits = nn.functional.interpolate(
            logits, size=(426,560), mode="bilinear", align_corners=False
        )
        upsampled_logits = upsampled_logits.squeeze(1)
        return upsampled_logits.exp()
    
    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        Y_prob = self(X)
        
        silog = self.criterionSILog(Y_prob, Y)
        sirme = self.criterionSIRME(Y_prob, Y)

        loss_task = silog if self.use_silog else sirme
        loss_orthogonal = 0
        loss_hoyer = 0
        for layer in self.decoder.linear_c:
            loss_orthogonal += layer.proj.orthogonal_loss()
            loss_hoyer += layer.proj.hoyer_loss()
            print(layer.proj.sigma)

        self.log('train_silog_loss', silog, prog_bar=True)
        self.log('train_sirme_loss', sirme, prog_bar=True)
        self.log('train_orthogonal_loss', loss_orthogonal, prog_bar=True)
        self.log('train_hoyer_loss', loss_hoyer, prog_bar=True)
        
        loss = loss_task + self.lambda_orth*loss_orthogonal + self.lambda_hoyer*self.lambda_hoyer
        return loss
    