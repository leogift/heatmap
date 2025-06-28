#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import time
import onnxruntime

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preproc(img, input_size, swap=(2, 0, 1)):
    assert(len(img.shape) == 3)
    padded_img = (np.ones((input_size[1], input_size[0], 3)) * 114).astype(np.uint8)
    print(input_size, img.shape)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def make_parser():
    parser = argparse.ArgumentParser("onnx inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="kptslite.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='wesine.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.7,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="kpts",
        help="Task type, kpts or mask.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # Create ONNX session
    session = onnxruntime.InferenceSession(args.model)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_shape = (input_shape[3], input_shape[2])

    rootdir = args.image_path
    mkdir(args.output_dir)
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            if filename[0] in [".", "_"]:
                continue
            if filename.split(".")[-1] not in ["jpg","JPG","bmp","BMP","png","PNG"]:
                continue

            origin_img = cv2.imread(os.path.join(parent, filename))
            img, ratio = preproc(origin_img, input_shape, (2, 0, 1))
            start_ts = time.time()

            output = session.run(None, {session.get_inputs()[0].name: img[None, :, :, :]})
            
            print("[Time]", (time.time()-start_ts))
            
            if args.task == "kpts":
                output = output[0]
            elif args.task == "mask":
                output = output[0][:, 1:]
            else:
                raise ValueError("Task type must be kpts or mask.")
            
            heatmap = (output.max(axis=1).squeeze()*255).astype(np.uint8)
            heatmap_path = os.path.join(args.output_dir, filename[:-4]+"_heatmap.png")
            cv2.imwrite(heatmap_path, heatmap)

            for i in range(0, output.shape[1]):
                heatmap = ((output[0, i]>args.score_thr).astype(np.uint8)).astype(np.uint8)*255
                heatmap_path = os.path.join(args.output_dir, filename[:-4]+"_heatmap_"+str(i)+".png")
                if args.task == "mask":
                    contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        contours = (contours * 4 / ratio).astype(int)
                        cv2.drawContours(origin_img, contours, 0, (0, 255, 0), 1)
                        contour_img_path = os.path.join(args.output_dir, filename[:-4]+"_contour_"+str(i)+".png")
                        cv2.imwrite(contour_img_path, origin_img)
                
                cv2.imwrite(heatmap_path, heatmap)
                    