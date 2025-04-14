import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from model_parts import *


class ResnetUnetDecoder(nn.Module):
    """
    Model with Resnet as encoder and unet as Decoder
    """
    def __init__(self, out_classes):
        super(ResnetUnetDecoder, self).__init__()
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
        self.outc = (OutConv(64, out_classes))

    def forward(self, x):
        with torch.no_grad():
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
        return logits
    
class ResnetTransformerUnet(nn.Module):
    """
    Model doing the following: Resnet+transformer+decoder.
    """
    def __init__(self, out_classes):
        super(ResnetTransformerUnet, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base_layers = list(self.resnet50.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        self.encoder_layer1 = nn.TransformerEncoderLayer(2048, 1, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(2048, 1, batch_first=True)
        self.encoder_layer3 = nn.TransformerEncoderLayer(2048, 1, batch_first=True)
        self.up1 = (DecoderUnetSkip(3072, 1024))
        self.up2 = (DecoderUnetSkip(1536, 512))
        self.up3 = (DecoderUnetSkip(768, 256))
        self.up4 = (DecoderUnetSkip(320, 128))
        self.up5 = (DecoderUnet(128, 64))
        self.outc = (OutConv(64, out_classes))
        
    def forward(self, x):
        with torch.no_grad():
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        tensor_heigth = x4.shape[2]
        tensor_width = x4.shape[3]
        x4 = torch.flatten(x4, start_dim=2)
        x4 = torch.transpose(x4, 1, 2)
        x4 = self.encoder_layer1(x4)
        x4 = self.encoder_layer2(x4)
        x4 = self.encoder_layer3(x4)
        x4 = torch.transpose(x4, 1, 2)
        print("Shape x4:", x4.shape)
        x4 = x4.reshape(x4.shape[0], x4.shape[1], tensor_heigth, tensor_width)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up5(x)
        logits = self.outc(x)
        logits = torch.exp(logits)
        return logits