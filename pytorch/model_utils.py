#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-22-23 15:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import torch
from torch.optim.lr_scheduler import MultiStepLR

from pytorch.lenet import LeNet5

__all__ = [
    "lenet5",
    "resnet18",
]

def build_model(model_name, dataset_name, **kwargs):
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
        if dataset_name == "mnist":
            # Modified for MNIST, for MNIST images are single-channel
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_filters = model.fc.in_features
            model.fc = torch.nn.Linear(num_filters, kwargs["output_dim"])

    return model

def build_scheduler(optimizer, model_name="resnet18"):
    """build_scheduler
    """
    scheduler = None

    if model_name not in __all__:
        raise ValueError(f"model_name ({model_name}) not in supported models: {__all__}.")
        
    if model_name == "resnet18":
        print("Using MultiStepLR.")
        scheduler = MultiStepLR(optimizer, milestones=[91,137,182], gamma=0.1)
        
    return scheduler