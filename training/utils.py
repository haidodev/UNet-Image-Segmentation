import torch
from training.model import UNet, UNetPlusPlus, AttentionUNet, DeepLabModel, TransUNet

def get_model(model_name):
    if model_name == "unet":
        return UNet()
    elif model_name == "unetpp":
        return UNetPlusPlus(base=32, deep_supervision=False)
    elif model_name == "attention_unet":
        return AttentionUNet()
    elif model_name == "deeplab":
        return DeepLabModel()
    elif model_name == "transunet":
        return TransUNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()