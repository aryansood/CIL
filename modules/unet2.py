import torch
from torch import nn
import torch.nn.functional as F

def match_size(dec, enc):
    return F.interpolate(dec, size=enc.shape[2:], mode="bilinear", align_corners=False)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.3):
        super(DoubleConv, self).__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Squeeze_Excite(nn.Module):
    def __init__(self,channel,reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    

class ExcitedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ExcitedDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            Squeeze_Excite(out_channels, 4),
        )

    def forward(self, x):
        return self.double_conv(x)



class UNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels=in_channels, out_channels=32, dropout_prob=dropout_prob)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(in_channels=32, out_channels=64, dropout_prob=dropout_prob)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(in_channels=64, out_channels=128, dropout_prob=dropout_prob)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(in_channels=128, out_channels=256, dropout_prob=dropout_prob)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(in_channels=256, out_channels=512, dropout_prob=dropout_prob)

        self.up4 = UpSample(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(in_channels=256 * 2, out_channels=256, dropout_prob=0.0)

        self.up3 = UpSample(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(in_channels=128 * 2, out_channels=128, dropout_prob=0.0)

        self.up2 = UpSample(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(in_channels=64 * 2, out_channels=64, dropout_prob=0.0)

        self.up1 = UpSample(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(in_channels=32 * 2, out_channels=32, dropout_prob=0.0)

        self.conv = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        enc4 = self.encoder4(self.down3(enc3))

        bottleneck = self.bottleneck(self.down4(enc4))
        
        dec4 = match_size(self.up4(bottleneck), enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = match_size(self.up3(dec4), enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = match_size(self.up2(dec3), enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = match_size(self.up1(dec2), enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

class SmallUNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels=in_channels, out_channels=16, dropout_prob=dropout_prob)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(in_channels=16, out_channels=32, dropout_prob=dropout_prob)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(in_channels=32, out_channels=64, dropout_prob=dropout_prob)

        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(in_channels=32 * 2, out_channels=32, dropout_prob=0.0)

        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(in_channels=16 * 2, out_channels=16, dropout_prob=0.0)

        self.conv = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))

        bottleneck = self.bottleneck(self.down2(enc2))
        
        dec2 = match_size(self.up2(bottleneck), enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = match_size(self.up1(dec2), enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))


if __name__ == '__main__':
    from torchinfo import summary

    model = UNetModule(1, 1)
    batch_size = 16
    summary(model, input_size=(batch_size, 1, 112, 112))