# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args):
    # 而build_vae则在act - main / detr / models / detr_vae.py中实现
    return build_vae(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)
