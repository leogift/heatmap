#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import argparse
from loguru import logger

import torch
from torch import nn

from heatmap.exp import get_exp
from heatmap.data import CustomMaskDataset, CustomKPTSDataset
from heatmap.utils import optimize_model

def make_parser():
    parser = argparse.ArgumentParser("HEATMAP onnx deploy")
    parser.add_argument(
        "-o", "--output-name", type=str, default="heatmap.onnx", help="output name of models"
    )
    parser.add_argument(
        "-s", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file)

    if exp.task=="kpts":
        valdataset = CustomKPTSDataset(
                data_dir=exp.data_dir,
                sequence_txt=exp.val_txt,
                image_size=exp.image_size,
                category_list=exp.category_list,
                augment=None,
            )
    elif exp.task=="mask":
        valdataset = CustomMaskDataset(
                data_dir=exp.data_dir,
                sequence_txt=exp.val_txt,
                image_size=exp.image_size,
                category_list=exp.category_list,
                augment=None,
            )
    image, mask, keypoints_xy, keypoints_visible = valdataset.pull_item(0)
    image = torch.tensor(image).unsqueeze(0).type(torch.float32)

    logger.info("loading dataset done.")

    model = exp.get_model()
    model = model.cpu()
    ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt, strict=True)

    input_args = [image]
    input_names = ["image"]

    model = optimize_model(model)

    model.eval()
    model.export = True

    logger.info("loading checkpoint done.")

    output_names = ["heatmap"]

    torch.onnx.export(
        model,
        tuple(input_args),
        args.output_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    import onnx
    from onnxsim import simplify

    # use onnx-simplifier to reduce reduent model.
    onnx_model = onnx.load(args.output_name)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_name)
    logger.info("generated simplified onnx model named {}".format(args.output_name))

if __name__ == "__main__":
    main()
