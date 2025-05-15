#!/usr/bin/env python3

import os
import torch
from torch import nn

from heatmap.exp import Exp as BaseExp

from loguru import logger

from heatmap.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

_CKPT_FULL_PATH = "weights/yolo_regnet_x_800mf_coco.pth"

class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.task = "kpts"

        self.data_dir = "coco-pose"

        self.train_txt = "train.txt"
        self.val_txt = "val.txt"

        self.image_size = (512, 512)  # (height, width)
        self.category_list = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                             'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                             'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        self.max_points_per_category = 2

        self.act = "relu"
        self.max_epoch = 120

        self.model_name = "regnet_x_800mf"

        self.warmup_epochs = 10
        self.no_aug_epochs = 10
        self.data_num_workers = 4
        self.eval_epoch_interval = 5


    def get_model(self):

        if "model" not in self.__dict__:
            from heatmap.models import HEATMAP, \
            BaseNorm, Regnet, YOLONeckFPN, RegnetNeckPAN, \
            HeatmapHead, AUXHeatmapHead, \
            C2aLayer, C2kLayer
            
            preproc = BaseNorm(trainable=True)

            pp_repeats = 0 if min(self.image_size[0], self.image_size[1])//32 <= 4 \
                else (min(self.image_size[0], self.image_size[1])//32 - 4)//6 + 1
            backbone = Regnet(
                    self.model_name,
                    act=self.act, 
                    pp_repeats=pp_repeats,
                    drop_rate=0.1,
                )
            self.channels = backbone.output_channels[-3:]
            neck = nn.Sequential(*[
                YOLONeckFPN(
                    in_channels=self.channels,
                    act=self.act,
                    layer_type=C2aLayer,
                    simple_reshape=True,
                    n=1
                ),
                RegnetNeckPAN(
                    self.model_name,
                    in_channels=self.channels,
                    act=self.act, 
                    layer_type=C2kLayer,
                    n=2
                ),
                YOLONeckFPN(
                    in_features=("pan3", "pan4", "pan5"),
                    in_channels=self.channels,
                    out_features=("feature3", "feature4", "feature5"),
                    act=self.act,
                    layer_type=C2aLayer,
                    simple_reshape=True,
                    n=1
                ),
            ])
            head = HeatmapHead(
                    in_feature="feature3",
                    in_channel=self.channels[0],
                    num_heatmap=len(self.category_list),
                    act=self.act, 
                    drop_rate=0.1,
                )

            aux_head_list = [
                    AUXHeatmapHead(
                        in_features=("fpn3", "fpn4", "fpn5"),
                        in_channels=self.channels,
                        num_heatmap=len(self.category_list)
                    ),
                    AUXHeatmapHead(
                        in_features=("pan3", "pan4", "pan5"),
                        in_channels=self.channels,
                        num_heatmap=len(self.category_list)
                    ),
                    AUXHeatmapHead(
                        in_features=("feature3", "feature4", "feature5"),
                        in_channels=self.channels,
                        num_heatmap=len(self.category_list)
                    ),
                ]
            
            self.model = HEATMAP(
                    preproc=preproc,
                    backbone=backbone,  
                    neck=neck, 
                    head=head, 
                    aux_head_list=aux_head_list,
                    task=self.task
                )
        
        ckpt = torch.load(_CKPT_FULL_PATH, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]

        for k in list(ckpt.keys()):
            if "pred" in k \
                or "loss" in k or "Loss" in k:
                del ckpt[k]

        incompatible = self.model.load_state_dict(ckpt, strict=False)
        logger.info("missing_keys:")
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )

        logger.info("unexpected_keys:")
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        
        return self.model
