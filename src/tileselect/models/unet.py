import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    """
    A lightweight U-Net implementation with configurable capacity and dropout.
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, base_ch=32, dropout=0.3, out_size=None):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # If set, the prediction is resized to (out_size, out_size) — decouples the
        # input resolution (e.g. 1024 thumbnail) from the output grid (e.g. 336 tiles).
        self.out_size = out_size

        c = base_ch  # 32 by default (was 64 → cuts params ~4x)
        self.inc = DoubleConv(n_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*2, c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*4, c*8))
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*8, c*16 // factor))
        
        self.dropout = nn.Dropout2d(p=dropout)
        
        self.up1 = Up(c*16, c*8 // factor, bilinear)
        self.up2 = Up(c*8, c*4 // factor, bilinear)
        self.up3 = Up(c*4, c*2 // factor, bilinear)
        self.up4 = Up(c*2, c, bilinear)
        self.outc = OutConv(c, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.dropout(x5)  # Regularize bottleneck
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.out_size is not None:
            logits = F.interpolate(logits, size=(self.out_size, self.out_size),
                                   mode='bilinear', align_corners=False)
        return logits
