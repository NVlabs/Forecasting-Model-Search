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

from collections import OrderedDict
import torch
import torch.nn as nn

from gmn.graph_construct.layers import Flatten, ImageTo1D, SinPosEnc, BasicBlock, PositionwiseFeedForward, SelfAttention, EquivSetLinear, TriplanarGrid, WeightEncodedImplicit

def make_small_cnn():
    # architecture of the small CNN from Unterthiner et al.
    model = nn.Sequential(nn.Conv2d(1, 16, 3),
                          nn.Conv2d(16, 16, 3),
                          nn.Conv2d(16, 16, 3),
                          nn.AdaptiveAvgPool2d((1,1)),
                          nn.Linear(16, 10))
    return model

def convert_state_dict(original_state_dict):
    new_state_dict = {}
    mapping = {
        'conv1.weight': '0.weight', 'conv1.bias': '0.bias',
        'conv2.weight': '2.weight', 'conv2.bias': '2.bias',
        'conv3.weight': '4.weight', 'conv3.bias': '4.bias',
        'fc.weight': '8.weight', 'fc.bias': '8.bias'
    }
    for key, value in original_state_dict.items():
        new_key = mapping.get(key)
        if new_key:
            new_state_dict[new_key] = value
    return new_state_dict

def sd_to_net(sd, architecture):
    if architecture == 'cnn':
        model = make_cnn(num_classes=10)
    elif architecture == 'simple_cnn':
        model = make_simple_cnn()
        sd = convert_state_dict(sd)
    elif architecture == 'resnet':
        model = make_resnet(num_classes=10)
    elif architecture == 'transformer':
        model = make_transformer(in_dim=3, hidden_dim=64, num_heads=2, out_dim=10, dropout=0.1, num_layers=2, vit=True, patch_size=4)
    elif architecture == 'deepsets':
        model = make_deepsets(num_classes=10)
    model.load_state_dict(sd)
    return model

def sd_to_triplanar_inr(sd, triplanar_res=32, triplanar_fdim=4, mlp_layers=[128, 64, 32, 1], spherical_bias=True, out_activation=None):
    model = WeightEncodedImplicit(mlp_layers, activation=nn.ReLU(), out_activation=None, triplanar_res=triplanar_res, triplanar_fdim=triplanar_fdim, spherical_bias=spherical_bias)
    model.load_state_dict(sd)
    return model



        
def make_simple_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(16, 10)
    )

def make_cnn(conv_layers=2, fc_layers=2, hidden_dim=32, in_dim=3, num_classes=10, activation='relu', norm='bn', dropout=0.0):
    layers = []
    activation_builder = {'relu': nn.ReLU,
                          'gelu': nn.GELU}[activation]
    norm_builder = {'bn': nn.BatchNorm2d,
                    'gn': lambda dim: nn.GroupNorm(2, dim),
                    'none': None}[norm]

    layers.append(nn.Conv2d(in_dim, hidden_dim, 3))
    layers.append(nn.Dropout2d(p=dropout))
    if norm_builder is not None: layers.append(norm_builder(hidden_dim))
    layers.append(activation_builder())

    for _ in range(conv_layers-1):
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3))
        layers.append(nn.Dropout2d(p=dropout))
        if norm_builder is not None: layers.append(norm_builder(hidden_dim))
        layers.append(activation_builder())

    layers.append(nn.AdaptiveAvgPool2d((1,1)))
    layers.append(Flatten())

    for _ in range(fc_layers-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_builder())
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)

def make_cnn_1d(conv_layers=2, fc_layers=2, hidden_dim=32, in_dim=3, num_classes=10, activation='relu', norm='bn', dropout=0.0):
    layers = []
    activation_builder = {'relu': nn.ReLU,
                          'gelu': nn.GELU}[activation]
    norm_builder = {'bn': nn.BatchNorm1d,
                    'gn': lambda dim: nn.GroupNorm(2, dim),
                    'none': None}[norm]

    layers.append(ImageTo1D())
    layers.append(nn.Conv1d(in_dim, hidden_dim, 9))
    layers.append(nn.Dropout1d(p=dropout)) # TODO: check if this behavior correct
    if norm_builder is not None: layers.append(norm_builder(hidden_dim))
    layers.append(activation_builder())

    for _ in range(conv_layers-1):
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, 9))
        layers.append(nn.Dropout1d(p=dropout))
        if norm_builder is not None: layers.append(norm_builder(hidden_dim))
        layers.append(activation_builder())

    layers.append(nn.AdaptiveAvgPool1d(1))
    layers.append(Flatten())

    for _ in range(fc_layers-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_builder())
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)

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

def make_resnet(conv_layers=2, hidden_dim=32, in_dim=3, num_classes=10, activation='relu'):
    layers = []
    activation_builder = nn.ReLU
    norm_builder = nn.BatchNorm2d

    layers.append(nn.Conv2d(in_dim, hidden_dim // 2, 5))
    if norm_builder is not None: layers.append(norm_builder(hidden_dim // 2))
    layers.append(activation_builder())

    layers.append(BasicBlock(hidden_dim // 2, hidden_dim))
    for _ in range(conv_layers-1):
        layers.append(BasicBlock(hidden_dim, hidden_dim))

    layers.append(nn.AdaptiveAvgPool2d((1,1)))
    layers.append(Flatten())

    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)


class Imageto1DSeq(nn.Module):
    # B x C x H x W -> B x H*W x C
    def forward(self, x): return x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)


def make_transformer(in_dim, hidden_dim, num_heads, out_dim, dropout=0.0, num_layers=2, vit=True, patch_size=4):
    layers = []

    if vit:
        # input B x C x H x W
        layers.append(nn.Conv2d(in_dim, hidden_dim, patch_size, stride=patch_size))
        layers.append(Imageto1DSeq())
        layers.append(SinPosEnc(hidden_dim))
    else:
        # input B x N x C
        layers.append(nn.Linear(in_dim, hidden_dim))

    for l in range(num_layers):
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(SelfAttention(hidden_dim, num_heads, dropout=dropout))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(PositionwiseFeedForward(hidden_dim, 4*hidden_dim, dropout=dropout))
    class AvgPoolSeq(nn.Module):
        # B x N x hidden_dim -> B x hidden_dim
        def forward(self, x): return x.mean(1)
    layers.append(AvgPoolSeq())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

