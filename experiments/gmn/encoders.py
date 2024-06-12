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
from gmn.graph_construct.constants import NODE_TYPES, EDGE_TYPES

class NodeEdgeFeatEncoder(nn.Module):
    def __init__(self, hidden_dim, norms=False, post_activation=True, ff=False, ff_scale=3, use_conv=True):
        super().__init__()

        self.norms = norms
        self.post_activation = post_activation
        self.use_conv = use_conv

        # node encoders
        if not ff:
            self.node_layer_encoder = nn.Sequential(nn.Linear(1, hidden_dim), Sin())
            self.neuron_num_encoder = nn.Sequential(nn.Linear(1, hidden_dim), Sin())
        else:
            self.node_layer_encoder = GaussianFourierFeatures(1, hidden_dim, scale=ff_scale)
            self.neuron_num_encoder = GaussianFourierFeatures(1, hidden_dim, scale=ff_scale)
        self.node_type_encoder = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.x_proj = nn.Linear(3*hidden_dim, hidden_dim)
        if norms:
            self.x_norm = nn.LayerNorm(3*hidden_dim)

        # edge encoders
        if not ff:
            self.weight_encoder = nn.Sequential(nn.Linear(1, hidden_dim), Sin())
            self.edge_layer_encoder = nn.Sequential(nn.Linear(1, hidden_dim), Sin())
            if use_conv: self.conv_pos_encoder = nn.Sequential(nn.Linear(3, hidden_dim), Sin())
        else:
            self.weight_encoder = GaussianFourierFeatures(1, hidden_dim, scale=ff_scale)
            self.edge_layer_encoder = GaussianFourierFeatures(1, hidden_dim, scale=ff_scale)
            if use_conv: self.conv_pos_encoder = GaussianFourierFeatures(3, hidden_dim, scale=ff_scale)

        self.edge_type_encoder = nn.Embedding(len(EDGE_TYPES), hidden_dim)
        edge_proj_dim = 4*hidden_dim if use_conv else 3*hidden_dim
        self.edge_attr_proj = nn.Linear(edge_proj_dim, hidden_dim)
        if norms:
            self.edge_attr_norm = nn.LayerNorm(edge_proj_dim)
        
        if post_activation:
            self.activation = nn.ReLU()

    def forward(self, x, edge_attr):
        x0 = self.node_layer_encoder(x[:, 0].unsqueeze(-1))
        x1 = self.neuron_num_encoder(x[:, 1].unsqueeze(-1))
        x2 = self.node_type_encoder(x[:, 2].long())
        x = torch.cat((x0, x1, x2), 1)
        if self.norms:
            x = self.x_norm(x)
        x = self.x_proj(x)
        

        e0 = self.weight_encoder(edge_attr[:, 0].unsqueeze(-1))
        e1 = self.edge_layer_encoder(edge_attr[:, 1].unsqueeze(-1))
        e2 = self.edge_type_encoder(edge_attr[:, 2].long())
        if self.use_conv:
            e3 = self.conv_pos_encoder(edge_attr[:, 3:])
            edge_attr = torch.cat((e0, e1, e2, e3), 1)
        else:
            edge_attr = torch.cat((e0, e1, e2), 1)
        if self.norms:
            edge_attr = self.edge_attr_norm(edge_attr)
        edge_attr = self.edge_attr_proj(edge_attr)

        if self.post_activation:
            x = self.activation(x)
            edge_attr = self.activation(edge_attr)

        return x, edge_attr

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_dim, out_dim, scale=3):
        super().__init__()
        self.scale = scale
        self.register_buffer('_weight', torch.randn((in_dim, out_dim//2)) * scale)

    def forward(self, x):
        x = x @ self._weight
        x = 2 * math.pi * x
        out = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return out
