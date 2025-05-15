#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from heatmap.utils import basic
from heatmap.models.losses import FocalLoss, HeatmapLoss, BalanceLoss, DiceLoss, UncertaintyLoss

# 模型框架
class HEATMAP(nn.Module):
    """
    HEATMAP model module.
    The network returns loss values during training
    and output results during test.
    """
    export = False

    def __init__(self, 
                preproc=None, # 数据预处理 basenorm/deepnorm
                backbone=None, # regnet
                neck=None, # fpn
                head=None, # bev head
                aux_head_list=[], # 辅助bev head
                task="kpts"
            ):
        super().__init__()

        # preproc
        self.preproc = nn.Identity() if preproc is None else preproc
        # backbone
        self.backbone = nn.Identity() if backbone is None else backbone
        # neck: fpn
        self.neck = nn.Identity() if neck is None else neck
        
        # occupancy head
        self.head = nn.Identity() if head is None else head
        # aux occupancy head
        self.aux_head_list = nn.ModuleList()
        for aux_head in aux_head_list:
            self.aux_head_list.append(aux_head)

        self.task = task

        # loss functions
        # preds vs targets
        self.bce_loss = HeatmapLoss(nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.13]), reduction="mean"))
        self.focal_loss = HeatmapLoss(FocalLoss(reduction="mean"))
        self.diff_loss = HeatmapLoss(BalanceLoss())
        self.dice_loss = HeatmapLoss(DiceLoss())

        # uncertainty loss
        self.bce_uncertainty_loss = UncertaintyLoss()
        self.focal_uncertainty_loss = UncertaintyLoss()
        self.diff_uncertainty_loss = UncertaintyLoss()
        self.dice_uncertainty_loss = UncertaintyLoss()

    def forward(self, image, mask=None, kpts_xy=None, kpts_visible=None):
        B, _, H, W = image.shape
        datas = {"input": image}
        datas = self.preproc(datas)
        datas = self.backbone(datas)
        datas = self.neck(datas)

        heatmap = self.head(datas)

        if self.training:
            # Training
            if self.task == "kpts":
                assert kpts_xy is not None and kpts_visible is not None, "points must be provided for training"
                # 用points制作heatmap
                radius = basic.gaussian_radius((H,W))
                heatmap_target \
                    = self.centermask(kpts_xy, kpts_visible, radius, (H, W)) # B, K, H, W
            elif self.task == "mask":
                assert mask is not None, "mask must be provided for training"
                heatmap_target = mask

            # main loss
            total_loss, bce_loss, focal_loss, diff_loss, dice_loss \
                = self.get_losses(heatmap, heatmap_target)
            
            # aux loss
            aux_preds_list = []
            for aux_head in self.aux_head_list:
                aux_preds = aux_head(datas)
                aux_preds_list.extend(aux_preds)
            
            aux_loss = 0
            for aux_preds in aux_preds_list:
                aux_total_loss, _, _, _, _ \
                    = self.get_losses(aux_preds, heatmap_target, uncertainty=False)
                aux_loss += 0.1 * aux_total_loss

            if len(aux_preds_list) > 0:
                aux_loss = aux_loss/len(aux_preds_list)
                total_loss += aux_loss

            # train outputs
            outputs = {
                "total_loss": total_loss,
                "bce_loss": bce_loss,
                "focal_loss": focal_loss,
                "diff_loss": diff_loss,
                "dice_loss": dice_loss,
                "aux_loss": aux_loss
            }

        elif not self.export:
            # Evaluation
            if self.task == "kpts":
                assert kpts_xy is not None and kpts_visible is not None, "points must be provided for evaluation"
                # 用points制作heatmap
                radius = basic.gaussian_radius((H,W))
                heatmap_target \
                    = self.centermask(kpts_xy, kpts_visible, radius, (H, W)) # B, K, H, W
            elif self.task == "mask":
                assert mask is not None, "mask must be provided for training"
                heatmap_target = mask

            similarity = self.get_similarity(heatmap, heatmap_target)
            dice = self.get_dice(heatmap, heatmap_target)

            # eval outputs
            outputs = {
                "similarity": similarity,
                "dice": dice,
            }

        else:
            # Export
            outputs = heatmap.sigmoid()
            
        return outputs
    
    # losses
    def get_losses(
        self,
        pred, # B, K, h, w
        target, # B, K, H, W
        uncertainty=True
    ):
        assert pred.shape[:2] == target.shape[:2], \
            f"expect {pred.shape[:2]} == {target.shape[:2]}"
        # 对齐
        Hp,Wp = pred.shape[2:]
        H,W = target.shape[2:]
        if Hp != H or Wp != W:
            _pred = F.interpolate(pred, (H,W))
        else:
            _pred = pred
        _target = target

        # losses
        bce_loss = self.bce_loss(_pred, _target.round())
        focal_loss = self.focal_loss(_pred, _target.round())
        diff_loss  = self.diff_loss(_pred.sigmoid(), _target)
        dice_loss = self.dice_loss(_pred.sigmoid().round(), _target.round())

        if uncertainty:
            # uncertainty weight
            if self.task == "kpts":
                diff_loss_weight = 5
                dice_loss_weight = 2
            elif self.task == "mask":
                diff_loss_weight = 2
                dice_loss_weight = 5

            bce_loss = self.bce_uncertainty_loss(bce_loss)
            focal_loss = self.bce_uncertainty_loss(focal_loss)
            diff_loss = self.diff_uncertainty_loss(diff_loss, diff_loss_weight)
            dice_loss = self.dice_uncertainty_loss(dice_loss, dice_loss_weight)

        total_loss = bce_loss + focal_loss \
            + diff_loss + dice_loss

        return total_loss, bce_loss, focal_loss, diff_loss, dice_loss

    # similarity
    def get_similarity(
        self,
        pred, # B, K, h, w
        target, # B, K, H, W
    ):
        assert pred.shape[:2] == target.shape[:2], \
            f"expect {pred.shape[:2]} == {target.shape[:2]}"
        # 对齐
        Hp,Wp = pred.shape[2:]
        H,W = target.shape[2:]
        if Hp != H or Wp != W:
            _pred = F.interpolate(pred, (H,W), mode="bilinear", align_corners=True)
        else:
            _pred = pred
        _target = target

        with torch.no_grad():
            similarity = 1 - self.diff_loss(_pred.sigmoid(), _target, debug=True)

        return similarity

    # dice
    def get_dice(
        self,
        pred, # B, K, h, w
        target, # B, K, H, W
    ):
        assert pred.shape[:2] == target.shape[:2], \
            f"expect {pred.shape[:2]} == {target.shape[:2]}"
        # 对齐
        Hp,Wp = pred.shape[2:]
        H,W = target.shape[2:]
        if Hp != H or Wp != W:
            _pred = F.interpolate(pred, (H,W), mode="bilinear", align_corners=True)
        else:
            _pred = pred
        _target = target

        with torch.no_grad():
            dice = 1 - self.dice_loss(_pred.sigmoid().round(), _target.round())

        return dice

    # 生成heatmap
    def centermask(self, kpts_xy, kpts_visible, radius, imgsz_HW):
        H, W = imgsz_HW
        B, C, _, _ = kpts_xy.shape # batch, category, kpts, 3

        kpts_xy = torch.round(kpts_xy)

        grid_h, grid_w = basic.meshgrid2d(1, H, W)
        grid_hw = torch.stack([grid_h, grid_w], dim=1).reshape(1, 1, 2, H, W)
        grid_hw = grid_hw.to(kpts_xy.device)

        batch_centermask = []
        for b in range(B):
            _centermask = []
            for c in range(C):
                _kpts_xy = kpts_xy[b, c]
                _kpts_visible = kpts_visible[b, c]
                # 从kpts_xy中取出可见的点
                _kpts_xy = _kpts_xy[_kpts_visible>0] # K,2
                if _kpts_xy.shape[0]>0:
                    mask_yx = torch.stack([_kpts_xy[:,1], _kpts_xy[:,0]], dim=1)
                    mask_yx = mask_yx.reshape(1, -1, 2, 1, 1) # 1,K,2,1,1

                    dist = grid_hw - mask_yx # 1,K,2,H,W
                    dist = torch.sum(dist**2, dim=2, keepdim=False)/(radius**2)

                    mask = torch.sqrt(torch.exp(-dist))
                    mask[mask < 0.01] = 0.0
                    mask = torch.max(mask, dim=1, keepdim=True)[0]

                else:
                    mask = torch.zeros((1, 1, H, W))

                mask = mask.to(kpts_xy.device)
                _centermask.append(mask)
                
            _centermask = torch.cat(_centermask, dim=1)
            batch_centermask.append(_centermask)

        batch_centermask = torch.cat(batch_centermask, dim=0)

        return batch_centermask
