#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    """LeNet-5 implemented with PyTorch
    LeNet5 的结构，是 conv2d x3 和 FC x2

    Args:
        padding: padding at the first Conv2d layer to change the input_shape of the element of the batched data. The inputs at first Conv2d layer must be (32, 32, 1) or (32, 32, 3), so set *padding* if necessary, e.g.*padding=2* for MNIST dataset.
        output_dim: number of top classifiers, e.g., 2, 10.

    References:
    - [1] https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    - [2] https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, padding=0, output_dim=10):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # conv1 pool
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # conv2 pool
        # flatten all dimensions except the batch dimension
        x = x.view(-1, x.size()[1:].numel())
        # x = torch.flatten(x, 1) is an alternative
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
