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
from typing import List, Optional
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.shape[0], -1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1, activation='relu'):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        
    def forward(self, x):
        return x + self.lin2(self.dropout(self.activation(self.lin1(x))))

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
    def forward(self, x):
        return x + self.attn(x, x, x, need_weights=False)[0]



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImageTo1D(nn.Module):
    def forward(self, x): return x.view(x.shape[0], x.shape[1], -1)


class SinPosEnc(nn.Module):
    # based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, dim, dim_last=True):
        super().__init__()
        self.dim = dim
        self.dim_last = dim_last
        self.freqs = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    def forward(self, x):
        # x: B x N x D if dim_last, else B x D x N
        if self.dim_last:
            seq_len = x.shape[1]
        else:
            seq_len = x.shape[2]
        idx = torch.arange(seq_len, device=x.device).unsqueeze(1).float() # N x 1
        freqs = self.freqs.to(idx) # 1 x D/2
        pe = torch.zeros(1, seq_len, self.dim, device=x.device)
        pe[0, :, 0::2] = torch.sin(idx * freqs) 
        pe[0, :, 1::2] = torch.cos(idx * freqs) 
        if not self.dim_last:
            pe = pe.transpose(2,1) # B x N x D -> B x D x N
        return x + pe


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1, activation='relu'):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return x + self.lin2(self.dropout(self.activation(self.lin1(x))))

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

class TriplanarGrid(nn.Module):
  def __init__(self, resolution: int, fdim: int):
    super().__init__()
    self._fdim = fdim
    tgrid = torch.empty(1, 3 * fdim, resolution, resolution)
    nn.init.xavier_uniform_(tgrid)
    self.tgrid = nn.Parameter(tgrid, requires_grad=True)

  def forward(self, x: torch.Tensor):
    """Trilinear interpolation of voxel grid features

    Args:
      x: torch.Tensor. Expects input `x` to have shape `[N,3]`. Should be in
         the range [-1,1]^3 (convention imposed by `F.grid_sample`).

    Returns:
      z: torch.Tensor. Has shape [N, K]
    """
    # To match the expected input shape of F.grid_sample
    xy = x[:,(0,1)].view(1, -1, 1, 2)
    yz = x[:,(1,2)].view(1, -1, 1, 2)
    zx = x[:,(2,0)].view(1, -1, 1, 2)

    tgridxy, tgridyz, tgridzx = torch.split(self.tgrid, self._fdim, 1)

    outxy = F.grid_sample(tgridxy, xy, align_corners=True).view(x.shape[0], -1)
    outyz = F.grid_sample(tgridyz, yz, align_corners=True).view(x.shape[0], -1)
    outzx = F.grid_sample(tgridzx, zx, align_corners=True).view(x.shape[0], -1)
    return torch.cat([x, outxy + outyz + outzx], -1)


class WeightEncodedImplicit(nn.Module):
    """Weight encoded implicit networks
    
    As described in https://arxiv.org/pdf/2009.09808.pdf
    """
    def __init__(
        self, mlp_layers: List[int], activation=nn.ReLU(),
        out_activation=nn.Tanh(), triplanar_res: Optional[int] = None,
        triplanar_fdim: Optional[int] = None, spherical_bias: bool = False
        ):
        super().__init__()
        self.n_layers = len(mlp_layers)

        layers = []
        indim = 3
        if triplanar_res is not None:
            layers.append(TriplanarGrid(triplanar_res, triplanar_fdim))
            indim = triplanar_fdim + 3
        layers.append(nn.Linear(indim, mlp_layers[0]))
        for i in range(self.n_layers - 1):
            layers.append(deepcopy(activation))
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
        # layers[-1].bias.data.zero_()
        self.layers = nn.Sequential(
            *layers
        )
        self.out_actvn = out_activation
        self.spherical_bias = spherical_bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network forward pass

        Args:
            x (torch.Tensor): Input coordinate (B, 3)

        Returns:
            torch.Tensor: Output sdf prediction (B, 1)
        """
        o = self.layers(x)
        if self.out_actvn:
            o = self.out_actvn(o)
        if self.spherical_bias:
            o += (x * x + 1e-10).sum(1, keepdim=True).sqrt_() - 0.5
        return o
