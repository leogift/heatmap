#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech Inc. All rights reserved.

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.image_size = (288, 512)  # (height, width)
        self.category_list = None
        self.task = "kpts" # kpts or mask
        # dir of dataset images
        self.data_dir = None
        # name of annotation file for training
        self.train_txt = "train.json"
        # name of annotation file for evaluation
        self.val_txt = "val.json"
        # max points per category
        self.max_points_per_category = 8

        # --------------- Augment ----------------- #
        self.no_aug = False
        # prob of applying hsv aug
        self.hsv_prob = 0.5
        # prob of applying blur aug
        self.blur_prob = 0.2
        # prob of applying erase aug
        self.erase_prob = 0.5
        # prob of applying mirror aug
        self.mirror_prob = 0
        # prob of applying affine aug
        self.affine_prob = 0.5
        # affine parameters
        self.affine_degrees = 45
        self.affine_translate = 0.1
        self.affine_scales = 0.2
        self.affine_shear = 5

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 10
        # max training epoch
        self.max_epoch = 240
        # minimum learning rate during warmup
        self.warmup_lr = 1e-6
        self.min_lr_ratio = 0.001
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 4e-3 / 64.0 # SGD: 0.04 / 64.0, Adam: 0.004 / 64.0
        # name of LRScheduler
        self.scheduler = "warmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 10
        # number of grad accumulation steps
        self.grad_accum = 4
        # apply EMA during training
        self.ema = True
        # Optimizer name
        self.opt_name = "AdamW"
        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_iter_interval = 50
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_epoch_interval = 5
        # save history checkpoint or not.
        # If set to False, yolo will only save latest and best ckpt.
        self.save_history_ckpt = False
        # Wether only train the head
        self.only_train_head = False
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        pass

    def get_train_loader(self, 
            batch_size, is_distributed
        ):
        from heatmap.data import (
            CustomKPTSDataset,
            CustomMaskDataset,
            DataAugment,
            AugmentBatchSampler,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from heatmap.utils import (
            wait_for_the_master,
            get_rank,
        )

        rank = get_rank()

        with wait_for_the_master(rank):
            if self.task=="kpts":
                traindataset = CustomKPTSDataset(
                    data_dir=self.data_dir,
                    sequence_txt=self.train_txt,
                    image_size=self.image_size,
                    category_list=self.category_list,
                    augment=DataAugment(
                        hsv_prob=self.hsv_prob,
                        blur_prob=self.blur_prob,
                        erase_prob=self.erase_prob,
                        mirror_prob=self.mirror_prob,
                        affine_prob=self.affine_prob,
                        degrees=self.affine_degrees,
                        translate=self.affine_translate,
                        scales=self.affine_scales,
                        shear=self.affine_shear,
                    ),
                    max_points_per_category=self.max_points_per_category
                )
            elif self.task=="mask":
                traindataset = CustomMaskDataset(
                    data_dir=self.data_dir,
                    sequence_txt=self.train_txt,
                    image_size=self.image_size,
                    category_list=self.category_list,
                    augment=DataAugment(
                        hsv_prob=self.hsv_prob,
                        blur_prob=self.blur_prob,
                        erase_prob=self.erase_prob,
                        mirror_prob=self.mirror_prob,
                        affine_prob=self.affine_prob,
                        degrees=self.affine_degrees,
                        translate=self.affine_translate,
                        scales=self.affine_scales,
                        shear=self.affine_shear,
                    ),
                )

        batch_size = batch_size // self.grad_accum
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(traindataset))

        batch_sampler = AugmentBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            aug=not self.no_aug,
            image_size=self.image_size
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        train_loader = torch.utils.data.DataLoader(traindataset, **dataloader_kwargs)

        return train_loader


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            g = [], [], []  # optimizer parameter groups
            norm = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
            for v in self.model.modules():
                for p_name, p in v.named_parameters(recurse=0):
                    if "bias" in p_name:
                        g[2].append(p)
                    elif p_name == "weight" and isinstance(v, norm):  # weight (no decay)
                        g[1].append(p)
                    else:
                        g[0].append(p)  # weight (with decay)

            if self.opt_name == "Adam":
                optimizer = torch.optim.Adam(g[2], lr=lr, betas=(self.momentum, 0.999))  # adjust beta1 to momentum
            elif self.opt_name == "AdamW":
                optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(self.momentum, 0.999), amsgrad=True)
            elif self.opt_name == "SGD":
                optimizer = torch.optim.SGD(g[2], lr=lr, momentum=self.momentum, nesterov=True)
            else:
                raise NotImplementedError(f"Optimizer {self.opt_name} not implemented.")

            optimizer.add_param_group({"params": g[0], "weight_decay": self.weight_decay})  # add g0 with weight_decay
            optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from heatmap.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def _get_val_loader(self, batch_size, is_distributed):
        from heatmap.data import (
            CustomKPTSDataset,
            CustomMaskDataset
        )

        if self.task=="kpts":
            valdataset = CustomKPTSDataset(
                data_dir=self.data_dir,
                sequence_txt=self.val_txt,
                image_size=self.image_size,
                category_list=self.category_list,
                augment=None,
                max_points_per_category=self.max_points_per_category
            )
        elif self.task=="mask":
            valdataset = CustomMaskDataset(
                data_dir=self.data_dir,
                sequence_txt=self.val_txt,
                image_size=self.image_size,
                category_list=self.category_list,
                augment=None,
            )

        batch_size = batch_size // self.grad_accum
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
            "drop_last": True,
        }
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed):
        from heatmap.evaluators import Evaluator

        val_loader = self._get_val_loader(batch_size, is_distributed)

        return Evaluator(dataloader=val_loader)

    def eval(self, model, evaluator):
        return evaluator.evaluate(model)
