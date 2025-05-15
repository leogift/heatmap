#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

# customDataset format:
'''
sequence txt file format:
single.png
...

single dict file format:
{
    "image": {
            "path": "20241121/image/1732154776703.jpg",
            "width": 1280,
            "height": 720
    },
    "mask": {
            "path": "20241121/mask/1732154776703.png",
            "width": 1280,
            "height": 720
    },
}
'''

import os
from loguru import logger

import cv2
import numpy as np

from heatmap.data.datasets import DatasetBase

from heatmap.data import get_dataset_root, data_preprocess

import json


class CustomMaskDataset(DatasetBase):
    """
    Custom dataset class.
    """
    def __init__(
        self,
        data_dir="custom",
        sequence_txt="train.txt",
        image_size=(288, 512),
        category_list=None,
        augment=None,
    ):
        """
        Args:
            data_dir (str): dataset root directory
            sequence_txt (str): sequence txt file name
            image_size (tuple[int]): image size
            augment: data augmentation strategy
        """
        super().__init__()
        self.data_dir = os.path.join(get_dataset_root(), data_dir)
        self.annotations_list = self._load_annotations(sequence_txt)
        self.image_size = image_size
        assert type(category_list)==list, "type(category_list) must be list"
        assert len(category_list)>0, "len(category_list) must greater than 0"
        self.category_list = category_list
        self.augment = augment

        self.datasets = self._preload_all()


    def __len__(self):
        return len(self.datasets)

    def _load_annotations(self, sequence_txt):
        # 读取sequence文件
        sequence = {}
        with open(os.path.join(self.data_dir, sequence_txt), "r") as f:
            sequence = f.readlines()
            f.close()

        # 读取单个标注文件
        annotations_list = []
        index = 0
        for single_json in sequence:
            single_json = single_json.replace('\r', '').replace('\n', '')
            if(len(single_json)<5):
                continue
            if single_json.split('.')[-1].lower() not in ["jpg", "png", "bmp"]:
                continue
            single_json = single_json[:-3] + "json"
            with open(os.path.join(self.data_dir, single_json), "r") as f:
                single_dict = json.load(f)
                annotations_list.append(single_dict)
                index += 1
                f.close()
        return annotations_list

    # 除图片，全载入内存
    def _load_camera_data(self, annotation):
        image = annotation["image"]

        return os.path.join(self.data_dir, image["path"])

    # 除点云，全载入内存
    def _load_mask_data(self, annotation):
        mask = annotation["mask"]

        return os.path.join(self.data_dir, mask["path"])


    def _load_by_index(self, index):
        annotation = self.annotations_list[index]

        imagefile = self._load_camera_data(annotation)
        maskfile = self._load_mask_data(annotation)

        return imagefile, maskfile


    def _preload_all(self):
        datasets = []
        for index in range(len(self.annotations_list)):
            dataset = self._load_by_index(index)
            datasets.append(dataset)
    
        return datasets


    # 拉取单个标注 读文件
    def pull_item(self, index, aug=False, image_size=None):
        imagefile, maskfile = self.datasets[index]

        # 读图
        image = cv2.imread(imagefile)
        mask = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)

        # 数据增强
        if aug and self.augment is not None:
                image, mask, _  = \
                    self.augment(image, mask, None)
        
        # 缩放至模型输入
        image, mask, _ = \
                data_preprocess(image, 
                                mask,
                                None,
                                image_size=self.image_size)
        
        # 多尺度训练
        if aug and image_size is not None and image_size != self.image_size:
                image, mask, _ = \
                    data_preprocess(image, 
                                    mask,
                                    None,
                                    image_size=image_size)

        # HWC -> CHW
        image = image.transpose(2, 0, 1).copy()

        mask_array = []
        for l in range(len(self.category_list)+1):
            mask_array.append((mask==l).astype(np.uint8))
        mask_array = np.stack(mask_array, axis=0)

        return image, mask_array, np.zeros(1), np.zeros(1)
