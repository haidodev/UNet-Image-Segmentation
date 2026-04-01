import torch
import torch.nn as nn
from model import UNet, UNetPlusPlus, AttentionUNet, DeepLabModel, TransUNet

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
    
def get_loss_function(loss_name):
    if loss_name == "bce":
        return bce_loss
    elif loss_name == "dice":
        return dice_loss
    elif loss_name == "bce_dice":
        return bce_dice_loss
    elif loss_name == "focal":
        return focal_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def bce_loss(pred, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(pred, target)

def bce_dice_loss(pred, target):
    bce = bce_loss(pred, target)
    d_loss = dice_loss(pred, target)
    return bce + d_loss

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred = torch.sigmoid(pred)
    bce = bce_loss(pred, target)
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def precision_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))

    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.mean()

def recall_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    tp = (pred * target).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))

    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.mean()