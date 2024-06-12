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
import torch.nn.functional as F

from dataset.pretrained_model_park.models.utils import Flatten, ImageTo1D, SinPosEnc

def make_deepsets(conv_layers=2, fc_layers=2, hidden_dim=32, in_dim=3, num_classes=10, activation='relu', norm='bn', dropout=0.0, pe=True):
    layers = []
    activation_builder = {'relu': nn.ReLU,
                          'gelu': nn.GELU}[activation]
    norm_builder = {'bn': nn.BatchNorm1d,
                    'gn': lambda dim: nn.GroupNorm(2, dim),
                    'none': None}[norm]

    layers.append(ImageTo1D())
    layers.append(nn.Conv1d(in_dim, hidden_dim, 1))
    layers.append(activation_builder())
    if pe:
        layers.append(SinPosEnc(hidden_dim, dim_last=False))
    layers.append(nn.Dropout1d(dropout))
    if norm_builder is not None: layers.append(norm_builder(hidden_dim))
    layers.append(activation_builder())

    for _ in range(conv_layers):
        layers.append(EquivSetLinear(hidden_dim, hidden_dim))
        layers.append(nn.Dropout1d(dropout))
        if norm_builder is not None: layers.append(norm_builder(hidden_dim))
        layers.append(activation_builder())

    layers.append(nn.AdaptiveAvgPool1d(1))
    layers.append(Flatten())

    for _ in range(fc_layers-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_builder())
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)

class EquivSetLinear(nn.Module):
    ''' Equivariant DeepSets linear layer 
        Input is B x D x N
        B is batch dim
        D is feature dim
        N is set size
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 1 x 1 convs for elementwise linear
        self.lin1 = nn.Conv1d(in_dim, out_dim, 1)
        self.lin2 = nn.Conv1d(in_dim, out_dim, 1, bias=False) # second bias redundant

    def forward(self, x):
        # x: B x C x H x W
        x1 = self.lin1(x)
        x2 = self.lin2(x.mean(2, keepdim=True))
        return x1 + x2
