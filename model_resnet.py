import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from unet_parts import *

class decoder_u(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class decoder_u_s(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x):
        x1 = self.up(x)
        return self.conv(x1)


class restnet_u(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(restnet_u, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base_layers = list(self.resnet50.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        self.up1 = (decoder_u(3072, 1024))
        self.up2 = (decoder_u(1536, 512))
        self.up3 = (decoder_u(768, 256))
        self.up4 = (decoder_u(320, 128))
        self.up5 = (decoder_u_s(128, 64))
        self.outc = (OutConv(64, n_classes))

        
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
        return logits
#         pass
# base_model = torchvision.models.resnet50(pretrained=True)

# base_layers = list(base_model.children())
# for i, layer in enumerate(base_model.children()):
#     print(i, layer)

