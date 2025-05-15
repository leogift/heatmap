#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hanqtech, Inc. and its affiliates.

import os
import random
import uuid

import numpy as np

import torch

import heatmap
import time

def get_dataset_root():
    heatmap_path = os.path.dirname(os.path.dirname(heatmap.__file__))
    return os.path.join(heatmap_path, "datasets")


def worker_init_reset_seed(worker_id):

    seed = (uuid.uuid4().int | int(time.time())) % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
