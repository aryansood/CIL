import torch
import torch.nn as nn
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
            Squeeze_Excite(out_channels, 4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x):
        return self.double_conv(x)



class UNetPlusPlusModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout_prob=0.2, deep_supervision=False, nb_filter = [32, 64, 128, 256, 512], excitation=False):
        super().__init__()
        ConvLayer = ExcitedDoubleConv if excitation else DoubleConv
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvLayer(in_channels, nb_filter[0], dropout_prob=dropout_prob)
        self.conv1_0 = ConvLayer(nb_filter[0], nb_filter[1], dropout_prob=dropout_prob)
        self.conv2_0 = ConvLayer(nb_filter[1], nb_filter[2], dropout_prob=dropout_prob)
        self.conv3_0 = ConvLayer(nb_filter[2], nb_filter[3], dropout_prob=dropout_prob)
        self.conv4_0 = ConvLayer(nb_filter[3], nb_filter[4], dropout_prob=dropout_prob)

        self.conv0_1 = ConvLayer(nb_filter[0]+nb_filter[1], nb_filter[0], dropout_prob=dropout_prob)
        self.conv1_1 = ConvLayer(nb_filter[1]+nb_filter[2], nb_filter[1], dropout_prob=dropout_prob)
        self.conv2_1 = ConvLayer(nb_filter[2]+nb_filter[3], nb_filter[2], dropout_prob=dropout_prob)
        self.conv3_1 = ConvLayer(nb_filter[3]+nb_filter[4], nb_filter[3], dropout_prob=dropout_prob)

        self.conv0_2 = ConvLayer(nb_filter[0]*2+nb_filter[1], nb_filter[0], dropout_prob=dropout_prob)
        self.conv1_2 = ConvLayer(nb_filter[1]*2+nb_filter[2], nb_filter[1], dropout_prob=dropout_prob)
        self.conv2_2 = ConvLayer(nb_filter[2]*2+nb_filter[3], nb_filter[2], dropout_prob=dropout_prob)

        self.conv0_3 = ConvLayer(nb_filter[0]*3+nb_filter[1], nb_filter[0], dropout_prob=dropout_prob)
        self.conv1_3 = ConvLayer(nb_filter[1]*3+nb_filter[2], nb_filter[1], dropout_prob=dropout_prob)

        self.conv0_4 = ConvLayer(nb_filter[0]*4+nb_filter[1], nb_filter[0], dropout_prob=dropout_prob)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, match_size(self.up(x1_0), x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, match_size(self.up(x2_0), x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, match_size(self.up(x1_1), x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, match_size(self.up(x3_0), x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, match_size(self.up(x2_1), x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, match_size(self.up(x1_2), x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, match_size(self.up(x4_0), x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, match_size(self.up(x3_1), x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, match_size(self.up(x2_2), x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, match_size(self.up(x1_3), x0_0)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return torch.exp(output)

if __name__ == "__main__":
    from torchinfo import summary

    in_channels = 3
    model = UNetPlusPlusModule(in_channels=in_channels, out_channels=1, excitation=True)
    batch_size = 1
    summary(model, input_size=(batch_size, in_channels, 560, 426))
    
    x = torch.randn((1, in_channels, 560, 426))
    preds = model(x.cuda())
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(preds)
