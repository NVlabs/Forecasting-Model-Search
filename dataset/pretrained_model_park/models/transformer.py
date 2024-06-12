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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.pretrained_model_park.models.utils import SinPosEnc

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1, activation='relu'):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return x + self.lin2(self.dropout(self.activation(self.lin1(x))))
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
    def forward(self, x):
        return x + self.attn(x, x, x, need_weights=False)[0]


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

