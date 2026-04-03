import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双重卷积块：Conv -> ReLU -> Conv -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样块：MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块：上采样 -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DEMCorrectionNet(nn.Module):
    """U-Net架构的DEM高程校正网络"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(DEMCorrectionNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # 编码器
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # 解码器
        self.up1 = Up(features[3] + features[2], features[2])
        self.up2 = Up(features[2] + features[1], features[1])
        self.up3 = Up(features[1] + features[0], features[0])
        
        # 输出层
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # 编码
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 解码
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # 输出
        x = self.outc(x)
        return x

# 简化版本：如果硬件资源有限
class SimpleDEMCorrectionNet(nn.Module):
    """轻量级DEM校正网络"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleDEMCorrectionNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        out = self.decoder(mid)
        return out