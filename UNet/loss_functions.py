import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc
