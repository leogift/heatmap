#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import cv2
import numpy as np
import random
import math

# 色彩抖动
# Hrange 42, Srange 212, Vrange 209
def augment_hsv(img, hgain=42/2, sgain=212/2, vgain=209/2):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)


# 运动模糊
def augment_blur(img, kernel=15, angle=180):
    kernel = abs(kernel)
    angle = abs(angle)

    # be sure the kernel size is odd
    kernel = round(np.random.randint(3, kernel))//2*2+1
    angle = np.random.uniform(-angle, angle)

    M = cv2.getRotationMatrix2D((kernel / 2, kernel / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel, kernel))
 
    motion_blur_kernel = motion_blur_kernel / kernel
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    # gaussian blur
    blurred = cv2.GaussianBlur(blurred, ksize=(kernel, kernel), sigmaX=0, sigmaY=0)

    return blurred


# 随机擦除
def augment_erase(img, ratio=0.2):
    ratio = abs(ratio)
    
    H,W = img.shape[:2]

    w = np.random.randint(3, round(W*ratio))
    h = np.random.randint(3, round(H*ratio))
    x = np.random.randint(0, W - w)
    y = np.random.randint(0, H - h)

    img[y:y+h, x:x+w] = 114
    return img

def augment_mirror(image, 
                   mask,
                   kpts, 
                   vertical=False, horizontal=False):
    height, width, _ = image.shape
    if horizontal:
        image = image[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]
        if kpts is not None:
            kpts[..., 0::2] = width - kpts[..., -2::-2]
    
    if vertical:
        image = image[::-1, :]
        if mask is not None:
            mask = mask[::-1, :]
        if kpts is not None:
            kpts[..., 1::2] = height - kpts[..., -1::-2]

    return image, mask, kpts


def get_affine_matrix(
    target_size,
    angle=45,
    translate=0.1,
    scale=0.2,
    shear=5,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = random.uniform(-math.fabs(angle), math.fabs(angle))
    scale = random.uniform(1.0-math.fabs(scale), 1.0+math.fabs(scale))

    R = cv2.getRotationMatrix2D(angle=angle, center=(twidth/2, theight/2), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(random.uniform(-math.fabs(shear), math.fabs(shear)) * math.pi / 180)
    shear_y = math.tan(random.uniform(-math.fabs(shear), math.fabs(shear)) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = random.uniform(-math.fabs(translate), math.fabs(translate)) * twidth  # x translation (pixels)
    translation_y = random.uniform(-math.fabs(translate), math.fabs(translate)) * theight  # y translation (pixels)

    M[0, 2] += translation_x
    M[1, 2] += translation_y

    return M


def apply_affine_to_kpts(kpts, M):
    tmp_kpts = np.ones((*kpts.shape[:-1], 3))
    tmp_kpts[..., :2] = kpts
    tmp_kpts = tmp_kpts @ M.T
    kpts = tmp_kpts[..., :2]
    return kpts

def augment_affine(
    image,
    mask,
    kpts,
    degrees=15,
    translate=0.1,
    scales=0.5,
    shear=5
):
    target_size = image.shape[1], image.shape[0]
    M = get_affine_matrix(target_size, degrees, translate, scales, shear)

    image = cv2.warpAffine(image, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if mask is not None:
        mask = cv2.warpAffine(mask, M, dsize=target_size, borderValue=(0, 0, 0))
    if kpts is not None:
        kpts = apply_affine_to_kpts(kpts, M)

    return image, mask, kpts


# 数据增强
class DataAugment:
    def __init__(self,
                hsv_prob=0,
                blur_prob=0,
                erase_prob=0,
                mirror_prob=0,
                affine_prob=0,
                degrees=45,
                translate=0.1,
                scales=0.2,
                shear=5
            ):

        self.hsv_prob = hsv_prob
        self.blur_prob = blur_prob
        self.erase_prob = erase_prob
        self.mirror_prob = mirror_prob
        self.affine_prob = affine_prob
        self.degrees = degrees
        self.translate = translate
        self.scales = scales
        self.shear = shear

    def __call__(self, 
                 input_image, 
                 input_mask=None,
                 input_kpts=None,
                ):

        image = input_image.copy()
        mask = None
        kpts = None
        if input_mask is not None:
            mask = input_mask.copy()
        if input_kpts is not None:
            kpts = input_kpts.copy()

        # 随机色彩抖动
        if np.random.random() < self.hsv_prob:
            image = augment_hsv(image)
        # 随机运动模糊
        if np.random.random() < self.blur_prob:
            image = augment_blur(image)
        # 随机擦除
        if np.random.random() < self.erase_prob:
            image = augment_erase(image)
        # 随机镜像
        if np.random.random() < self.mirror_prob:
            image, mask, kpts = augment_mirror(image, 
                                         mask,
                                         kpts, 
                                         horizontal=True)
        if np.random.random() < self.mirror_prob:
            image, mask, kpts = augment_mirror(image,
                                         mask, 
                                         kpts, 
                                         vertical=True)
        # 随机仿射
        if np.random.random() < self.affine_prob:
            image, mask, kpts = augment_affine(image, 
                    mask,
                    kpts,
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scales,
                    shear=self.shear
                )

        return image, mask, kpts
