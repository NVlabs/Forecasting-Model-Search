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

from torch_scatter import scatter

class GNNwEdgeReadout(nn.Module):
    def __init__(self, gnn, readout, use_nodes=False):
        super().__init__()
        self.gnn = gnn
        self.readout = readout
        self.use_nodes = use_nodes

    def forward(self, x, edge_index, edge_attr, batch):
        x, edge_attr = self.gnn(x, edge_index, edge_attr, None, batch)
        if self.use_nodes:
            graph_feat = self.readout(x, edge_index, edge_attr, batch)
        else:
            graph_feat = self.readout(edge_index, edge_attr, batch)
        return graph_feat



################################
###### Pooling Operations ######
################################

class BasicNodePool(nn.Module):
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce
    
    def forward(self, x, batch, **kwargs):
        return scatter(x, batch, dim=0, reduce=self.reduce)

class DSNodeEdgeReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, reduce='mean'):
        super().__init__()
        self.node_pool = BasicNodePool(reduce)
        self.edge_pool = BasicEdgePool(reduce)
        self.pre_pool_x = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.pre_pool_e = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        layers = [nn.LayerNorm(2*hidden_dim),
                  nn.Linear(2*hidden_dim, hidden_dim)]
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.post_pool = nn.Sequential(*layers)
    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        x = self.pre_pool_x(x)
        edge_attr = self.pre_pool_e(edge_attr)

        graph_feat_x = self.node_pool(x, batch)
        graph_feat_e = self.edge_pool(edge_index, edge_attr, batch)
        graph_feat = torch.cat([graph_feat_x, graph_feat_e], dim=-1)
        return self.post_pool(graph_feat)

# Basic pooling, e.g. summation over all edges in the graph.

class BasicEdgePool(nn.Module):
    # sums, averages, or max pools over all edge features
    # to get graph-level representation
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce
    
    def forward(self, edge_index, edge_attr, batch, **kwargs):
        edge_batch = batch[edge_index[0]]
        return scatter(edge_attr, edge_batch, dim=0, reduce=self.reduce)
    
class MLPEdgeReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, reduce='mean'):
        super().__init__()
        self.pool = BasicEdgePool(reduce)
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, edge_index, edge_attr, batch, **kwargs):
        graph_feat = self.pool(edge_index, edge_attr, batch)
        return self.mlp(graph_feat)

class DSEdgeReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, reduce='mean'):
        super().__init__()
        self.pool = BasicEdgePool(reduce)
        self.pre_pool = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        layers = [nn.Linear(hidden_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.post_pool = nn.Sequential(*layers)
    def forward(self, edge_index, edge_attr, batch, **kwargs):
        edge_attr = self.pre_pool(edge_attr)
        graph_feat = self.pool(edge_index, edge_attr, batch)
        return self.post_pool(graph_feat)
