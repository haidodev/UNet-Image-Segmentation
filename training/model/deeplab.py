import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = deeplabv3_resnet50(
            weights=None,
            num_classes=1
        )

    def forward(self, x):
        out = self.model(x)["out"]
        return out