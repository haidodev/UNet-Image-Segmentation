import torch 
import torch.nn as nn
from .double_convolution import DoubleConv
    
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, base=32, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv00 = DoubleConv(in_channels, base)
        self.conv10 = DoubleConv(base, base*2)
        self.conv20 = DoubleConv(base*2, base*4)
        self.conv30 = DoubleConv(base*4, base*8)
        self.conv40 = DoubleConv(base*8, base*16)
        
        self.conv01 = DoubleConv(base + base*2, base)
        self.conv11 = DoubleConv(base*2 + base*4, base*2)
        self.conv21 = DoubleConv(base*4 + base*8, base*4)
        self.conv31 = DoubleConv(base*8 + base*16, base*8)
        
        self.conv02 = DoubleConv(base + base + base*2, base)
        self.conv12 = DoubleConv(base*2 + base*2 + base*4, base*2)
        self.conv22 = DoubleConv(base*4 + base*4 + base*8, base*4)
        
        self.conv03 = DoubleConv(base + base + base + base*2, base)
        self.conv13 = DoubleConv(base*2 + base*2 + base*2 + base*4, base*2)
        
        self.conv04 = DoubleConv(base + base + base + base + base*2, base)
        
        if deep_supervision:
            self.final1 = nn.Conv2d(base, 1, 1)
            self.final2 = nn.Conv2d(base, 1, 1)
            self.final3 = nn.Conv2d(base, 1, 1)
            self.final4 = nn.Conv2d(base, 1, 1)
        else:
            self.final = nn.Conv2d(base, 1, 1)
            
            
    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))
        
        x01 = self.conv01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up(x40)], dim=1))
        
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up(x31)], dim=1))
        
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up(x22)], dim=1))
        
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up(x13)], dim=1))
        
        if self.deep_supervision:
            return (self.final1(x01), 
                    self.final2(x02), 
                    self.final3(x03), 
                    self.final4(x04))
        else:
            return self.final(x04)
        
