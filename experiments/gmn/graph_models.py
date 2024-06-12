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
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, use_global=False):
        super().__init__()
        self.use_global = use_global
        assert not use_global, 'global not yet implemented'
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, reduce='mean', use_global=False):
        super().__init__()
        self.reduce = reduce
        self.use_global = use_global
        assert not use_global, 'global not yet implemented'
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce=self.reduce)
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
        
class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, reduce='mean'):
        super().__init__()
        self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u,
            scatter(x, batch, dim=0, reduce=self.reduce)], dim=1)
        return self.global_mlp(out)

class EdgeMPNN(nn.Module):
    # in the sense of Battaglia et al.
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, node_out_dim, edge_out_dim, num_layers, use_bn=True, dropout=0.0, reduce='mean', global_in_dim=0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce

        if num_layers == 1:
            edge_model = EdgeModel(edge_in_dim + 2*node_in_dim, edge_out_dim, activation=False)
            self.convs.append(MetaLayer(edge_model=edge_model))
        else:
            edge_model = EdgeModel(edge_in_dim+node_in_dim*2, hidden_dim) 
            node_model = NodeModel(node_in_dim+hidden_dim, hidden_dim, reduce=self.reduce)
            #global_model = 
            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers-2):
                edge_model = EdgeModel(3*hidden_dim, hidden_dim) 
                node_model = NodeModel(2*hidden_dim, hidden_dim, reduce=self.reduce)
                self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model))
                self.node_norms.append(nn.BatchNorm1d(hidden_dim))
                self.edge_norms.append(nn.BatchNorm1d(hidden_dim))

            edge_model = EdgeModel(3*hidden_dim, edge_out_dim, activation=False) 
            node_model = NodeModel(hidden_dim+edge_out_dim, node_out_dim, activation=False, reduce=self.reduce)
            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model))

    def forward(self, x, edge_index, edge_attr, *args):
        for i, conv in enumerate(self.convs):
            x, edge_attr, _ = conv(x, edge_index, edge_attr)
            if i != len(self.convs)-1 and self.use_bn:
                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
        return x, edge_attr

class ResEdgeMPNNBlock(nn.Module):
    # inspired by DiT block
    # does not update global variable right now
    def __init__(self, hidden_dim, dropout=0.0, reduce='mean', activation='silu', update_node=True, use_global=False, update_edge=True):
        super().__init__()

        assert not use_global, 'global feat not implemented'
        self.reduce = reduce
        self.update_node = update_node
        self.update_edge = update_edge
        self.dropout = dropout

        if activation == 'relu':
            self.activation_builder = nn.ReLU
        elif activation == 'silu':
            self.activation_builder = nn.SiLU
        else:
            raise ValueError('Invalid activation')

        edge_model = EdgeModel(3*hidden_dim, hidden_dim, use_global=use_global) if self.update_edge else None
        node_model = NodeModel(2*hidden_dim, hidden_dim, reduce=self.reduce, use_global=use_global) if self.update_node else None
        global_model = None
            
        self.conv = MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model)
        self.node_norm = nn.LayerNorm(hidden_dim) if self.update_node else None
        self.edge_norm = nn.LayerNorm(hidden_dim) if self.update_edge else None

        self.node_mlp = nn.Sequential(self.activation_builder(), nn.Linear(hidden_dim, hidden_dim)) if self.update_node else None
        self.edge_mlp = nn.Sequential(self.activation_builder(), nn.Linear(hidden_dim, hidden_dim)) if self.update_edge else None
       
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: num_nodes x hidden_dim
        # edge_index: 2 x num_edges
        # edge_attr: num_edges x hidden_dim
        # u: batch_size x hidden_dim
        # batch: num_nodes

        orig_x, orig_edge_attr = x, edge_attr
        if self.update_node:
            x = self.node_norm(x)
        if self.update_edge:
            edge_attr = self.edge_norm(edge_attr)

        x, edge_attr, _ = self.conv(x, edge_index, edge_attr, u=None, batch=None)

        if self.update_node:
            x = F.dropout(x, self.dropout, self.training)
            x = self.node_mlp(x)
            x = orig_x + x

        if self.update_edge:
            edge_attr = F.dropout(edge_attr, self.dropout, self.training)
            edge_attr = self.edge_mlp(edge_attr)
            edge_attr = orig_edge_attr + edge_attr

        return x, edge_attr

class EdgeMPNNDiT(nn.Module):
    ''' Structure vaguely inspired by DiT: "Scalable Diffusion Models with Transformers" by Peebles and Xie'''
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, edge_out_dim, num_layers,  dropout=0.0, reduce='mean', activation='silu', use_global=False, **kwargs):
        super().__init__()
        assert not use_global, 'global feat not implemented'

        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers-1):
            block = ResEdgeMPNNBlock(hidden_dim, dropout=dropout, reduce=reduce, activation=activation, update_node=True, use_global=False, update_edge=True)
            self.convs.append(block)

        block = ResEdgeMPNNBlock(hidden_dim, dropout=dropout, reduce=reduce, activation=activation, update_node=False, use_global=False)
        self.convs.append(block)
        self.final_layer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, edge_out_dim))
    
    def init_weights_(self):
        # identity init for each block
        for block in self.convs:
            if block.update_node:
                nn.init.constant_(block.node_mlp[-1].weight, 0)
                nn.init.constant_(block.node_mlp[-1].bias, 0)

            if block.update_edge:
                nn.init.constant_(block.edge_mlp[-1].weight, 0)
                nn.init.constant_(block.edge_mlp[-1].bias, 0)

        # make output zero at start
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def forward(self, x, edge_index, edge_attr, u, batch):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for block in self.convs:
            x, edge_attr = block(x, edge_index, edge_attr, u, batch)
        edge_attr = self.final_layer(edge_attr)
        return x, edge_attr

