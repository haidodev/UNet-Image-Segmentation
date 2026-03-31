import torch 
import torch.nn as nn
from .double_convolution import DoubleConv
from .encoder import Encoder
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)

        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.pos_embed = None
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        if self.pos_embed is None or self.pos_embed.shape[1] != H * W:
            self.pos_embed = nn.Parameter(torch.zeros(1, H * W, C)).to(x.device)
    
        x = x + self.pos_embed

        
        h = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = x + h
        
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
    
class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        
        self.transformer = nn.Sequential(
            TransformerBlock(512, num_heads=8),
            TransformerBlock(512, num_heads=8),
        )
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        
        t = self.transformer(e4)
        
        d4 = self.up4(t)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)        
        