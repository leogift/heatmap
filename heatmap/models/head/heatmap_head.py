#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn

from heatmap.models.network_blocks import get_activation, BaseConv
from heatmap.utils import initialize_weights, special_multiples

class HeatmapHead(nn.Module):
    def __init__(
        self,
        in_feature="fpn3",
        in_channel=256,
        num_heatmap=4,
        act="silu",
        simple_reshape=False,
        drop_rate=0.,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
        """
        super().__init__()
        self.in_feature = in_feature

        # 主输出头
        self.upsample = nn.Sequential(*[
			nn.ConvTranspose2d(
	            int(in_channel), 
	            int(in_channel), 
	            kernel_size=4, 
	            stride=2, 
	            padding=1,
	            groups=int(in_channel),
	            bias=True),
			get_activation(act)()
		]) if simple_reshape==False else nn.Upsample(scale_factor=2, mode="bilinear")

        hidden_channel = special_multiples(in_channel/num_heatmap, 8)
        self.stems = nn.ModuleList(
            BaseConv(
                in_channels=int(in_channel),
                out_channels=hidden_channel,
                ksize=1,
                stride=1,
                act=act,
            )
            for _ in range(num_heatmap+1)
        )

        self.convs = nn.ModuleList(
            nn.Sequential(*[
                BaseConv(
                    in_channels=hidden_channel,
                    out_channels=hidden_channel,
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                BaseConv(
                    in_channels=hidden_channel,
                    out_channels=hidden_channel,
                    ksize=3,
                    stride=1,
                    act=act,
                )
            ])
            for _ in range(num_heatmap+1)
        )

        self.preds = nn.ModuleList(
            nn.Conv2d(
                hidden_channel, 
                1,
                kernel_size=1, 
                bias=True)
            for _ in range(num_heatmap+1)
        )

        initialize_weights(self)
 
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()


    def forward(self, inputs):
        x = inputs[self.in_feature]

        y = self.upsample(x)
        outputs = []
        for stem, conv, pred in zip(self.stems, self.convs, self.preds):
            _x = stem(x)
            _x = conv(_x)
            if self.training:
                _x = self.drop(_x)
            _x = pred(_x)
            outputs.append(_x)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
