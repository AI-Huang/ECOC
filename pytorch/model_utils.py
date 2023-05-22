#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-22-23 15:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

from pytorch.lenet import LeNet5

__all__ = [
    "lenet5",
    "resnet18",
]

def build_model(model_name, **kwargs):
    """build_model
    Input:
        model_name: string, 

    Return:

    """
    if model_name not in __all__:
        raise ValueError(f"model_name ({model_name}) not in supported models: {__all__}.")

    if model_name == "lenet5":
        model = LeNet5(**kwargs)

    elif model_name == "resnet18":
        from torchvision.models.resnet import resnet18
        model = resnet18()

    return model
