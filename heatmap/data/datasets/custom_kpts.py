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
            "path": "20241121/cameras/front/1732154776703.jpg",
            "width": 1280,
            "height": 720
    },
    "keypoints": {
        "bundle": [
                {
                    "x": 1,
                    "y": 2,
                    "visible": 1,
                },
                ...
            ],
        "apex": [
                {
                    "x": 1,
                    "y": 2,
                    "visible": 1,
                },
                ...
            ],
        "head": [
                {
                    "x": 1,
                    "y": 2,
                    "visible": 1,
                },
                ...
            ],
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


class CustomKPTSDataset(DatasetBase):
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
        max_points_per_category=1,
    ):
        """
        Args:
            data_dir (str): dataset root directory
            sequence_txt (str): sequence txt file name
            image_size (tuple[int]): image size
            category_list (list[str]): list of category names
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
        self.max_points_per_category = max_points_per_category

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
    def _load_keypoints_data(self, annotation):
        keypoints_xy_list = []
        keypoints_visible_list = []

        keypoints = annotation["keypoints"]
        for category in self.category_list:
            keypoints_xy = []
            keypoints_visible = []
            if category in keypoints:
                for point in keypoints[category]:
                    x = point["x"]
                    y = point["y"]
                    visible = point["visible"]
                    keypoints_xy.append([x, y])
                    keypoints_visible.append(visible)
            else:
                keypoints_xy.append([-1, -1])
                keypoints_visible.append(0)
            
            keypoints_xy_list.append(keypoints_xy)
            keypoints_visible_list.append(keypoints_visible)
        
        return keypoints_xy_list, keypoints_visible_list


    def _load_by_index(self, index):
        annotation = self.annotations_list[index]

        imagefile = self._load_camera_data(annotation)
        keypoints_xy_list, keypoints_visible_list = self._load_keypoints_data(annotation)

        return imagefile, keypoints_xy_list, keypoints_visible_list


    def _preload_all(self):
        datasets = []
        for index in range(len(self.annotations_list)):
            dataset = self._load_by_index(index)
            datasets.append(dataset)
    
        return datasets


    # 拉取单个标注 读文件
    def pull_item(self, index, aug=False, image_size=None):
        imagefile, keypoints_xy_list, keypoints_visible_list = self.datasets[index]

        # 读图
        image = cv2.imread(imagefile)

        # 读点
        keypoints_xy_array = []
        keypoints_visible_array = []
        for keypoints_xy, keypoints_visible in zip(keypoints_xy_list, keypoints_visible_list):
            keypoints_xy = np.array(keypoints_xy) # Nx2
            keypoints_visible = np.array(keypoints_visible) # Nx1
            # 对齐max_lidar_points个点
            if keypoints_xy.shape[0] > self.max_points_per_category:
                inds = np.random.choice(keypoints_xy.shape[0], self.max_points_per_category, replace=False)
                keypoints_xy = keypoints_xy[inds]
                keypoints_visible = keypoints_visible[inds]
            elif keypoints_xy.shape[0] > 0:
                inds = np.random.choice(keypoints_xy.shape[0], self.max_points_per_category, replace=True)
                keypoints_xy = keypoints_xy[inds]
                keypoints_visible = keypoints_visible[inds]
            else:
                keypoints_xy = np.zeros((self.max_points_per_category, 2))
                keypoints_visible = np.zeros((self.max_points_per_category))
            
            keypoints_xy_array.append(keypoints_xy)
            keypoints_visible_array.append(keypoints_visible)
        
        keypoints_xy_array = np.stack(keypoints_xy_array)
        keypoints_visible_array = np.stack(keypoints_visible_array)
        
        # 数据增强
        if aug and self.augment is not None:
                image, _, keypoints_xy_array  = \
                    self.augment(image, None, keypoints_xy_array)
        
        # 缩放至模型输入
        image, _, keypoints_xy_array = \
                data_preprocess(image, 
                                None,
                                keypoints_xy_array, 
                                image_size=self.image_size)
        
        # 多尺度训练
        if aug and image_size is not None and image_size != self.image_size:
                image, _, keypoints_xy_array = \
                    data_preprocess(image, 
                                    None,
                                    keypoints_xy_array, 
                                    image_size=image_size)

        # HWC -> CHW
        image = image.transpose(2, 0, 1).copy()

        return image, np.zeros(1), keypoints_xy_array, keypoints_visible_array
