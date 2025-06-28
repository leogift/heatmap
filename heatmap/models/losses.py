#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in [ None, 'none', 'mean', 'sum']
        self.reduction = reduction

    '''
    Args:
        pred: tensor without sigmoid
        target: tensor
    '''
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            loss = torch.ones([1, target.shape[1:]], device=pred.device) * target.shape[1:]

        else:
            pred_sigmoid = pred.sigmoid()
            target = target.type_as(pred)
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            weight = pt.pow(self.gamma)
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            weight = alpha_factor * weight
            loss = weight * F.binary_cross_entropy_with_logits(
                pred, target, 
                reduction="none")

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss

# uncertainty loss
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, loss, factor=1):
        return F.relu(factor*torch.exp(-self.weight)*loss + self.weight)

# heatmap loss
class HeatmapLoss(nn.Module):
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss(reduction="mean")):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, debug=False):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"
        
        target = target.type_as(pred)

        B, C, H, W = pred.shape
        loss = 0
        for c in range(C):
            loss += self.loss_fn(pred[:, c], target[:, c])

            if debug: 
                import cv2
                import numpy as np
                import os
                if not os.path.exists("debug"):
                    os.mkdir("debug")
                heatmap_pred = (pred[0, c].detach().cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(f"debug/heatmap_pred_{c}.png", heatmap_pred)
                heatmap_target = (target[0, c].cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(f"debug/heatmap_target_{c}.png", heatmap_target)
                
        return loss / C

# balance loss
class BalanceLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss(reduction="none")):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = self.loss_fn(pred, target)

        g_mask = target.gt(0.5).float()
        l_mask = target.lt(0.5).float()

        g_loss = (loss * g_mask).sum() / g_mask.sum().clamp(1e-7)
        l_loss = (loss * l_mask).sum() / l_mask.sum().clamp(1e-7)

        loss = (g_loss + l_loss) * 0.5

        return loss

# dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    '''
    Args:
        pred: tensor in Integer
        target: tensor in Integer
    '''
    def forward(self, pred, target):
        assert pred.shape == target.shape, \
            f"expect {pred.shape} == {target.shape}"

        target = target.type_as(pred)

        loss = 1 - 2 * (pred * target).sum() / (pred + target).sum().clamp(1e-7)

        return loss
