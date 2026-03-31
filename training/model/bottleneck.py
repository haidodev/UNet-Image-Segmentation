import torch.nn as nn
from .double_convolution import DoubleConv

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(512, 1024)

    def forward(self, e4):
        x = self.pool(e4)
        x = self.conv(x)
        return x