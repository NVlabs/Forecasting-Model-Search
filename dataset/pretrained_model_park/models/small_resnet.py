# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
from dataset.pretrained_model_park.models.resnet import BasicBlock
from dataset.pretrained_model_park.models.utils import Flatten

def conv3x3(in_dim, out_dim, stride=1):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)

def _make_resnet_block(block, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(in_planes, out_planes, stride))
    return layers


def make_sequential_resnet(num_blocks, dim=32, num_classes=10):
    # only works for basic block
    curr_dim = dim
    layers = []
    layers.append(conv3x3(3, dim, stride=1))
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    layers.extend(_make_resnet_block(BasicBlock, dim, dim, num_blocks[0], stride=1))
    layers.extend(_make_resnet_block(BasicBlock, dim, 2*dim, num_blocks[1], stride=2))
    layers.extend(_make_resnet_block(BasicBlock, 2*dim, 4*dim, num_blocks[2], stride=2))
    layers.extend(_make_resnet_block(BasicBlock, 4*dim, 8*dim, num_blocks[3], stride=2))
    layers.append(nn.AvgPool2d(4))
    layers.append(Flatten())
    layers.append(nn.Linear(8*dim, num_classes))

    return nn.Sequential(*layers)
    
