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
from copy import deepcopy
import torch
import torch.nn as nn
import torch_geometric

from gmn.graph_construct.constants import NODE_TYPES, EDGE_TYPES, CONV_LAYERS, NORM_LAYERS, RESIDUAL_LAYERS
from gmn.graph_construct.utils import make_node_feat, make_edge_attr, conv_to_graph, linear_to_graph, norm_to_graph, ffn_to_graph, basic_block_to_graph, self_attention_to_graph, equiv_set_linear_to_graph, triplanar_to_graph
from gmn.graph_construct.layers import Flatten, PositionwiseFeedForward, BasicBlock, SelfAttention, EquivSetLinear, TriplanarGrid

# model <--> arch <--> graph

def sequential_to_arch(model):
    # input can be a nn.Sequential
    # or ordered list of modules
    arch = []
    weight_bias_modules = CONV_LAYERS + [nn.Linear] + NORM_LAYERS
    for module in model:
        layer = [type(module)]
        if type(module) in weight_bias_modules:
            layer.append(module.weight)
            layer.append(module.bias)
        elif type(module) == BasicBlock:
            layer.extend([
                module.conv1.weight,
                module.bn1.weight,
                module.bn1.bias,
                module.conv2.weight,
                module.bn2.weight,
                module.bn2.bias])
            if len(module.shortcut) > 0:
                layer.extend([
                    module.shortcut[0].weight,
                    module.shortcut[1].weight,
                    module.shortcut[1].bias])
        elif type(module) == PositionwiseFeedForward:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
            layer.append(module.lin2.bias)
        elif type(module) == SelfAttention:
            layer.append(module.attn.in_proj_weight)
            layer.append(module.attn.in_proj_bias)
            layer.append(module.attn.out_proj.weight)
            layer.append(module.attn.out_proj.bias)
        elif type(module) == EquivSetLinear:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
        elif type(module) == TriplanarGrid:
            layer.append(module.tgrid)
        else:
            if len(list(module.parameters())) != 0:
                raise ValueError(f'{type(module)} has parameters but is not yet supported')
            continue
        arch.append(layer)
    return arch

def arch_to_graph(arch, self_loops=False):
    
    curr_idx = 0
    x = []
    edge_index = []
    edge_attr = []
    layer_num = 0
    
    # initialize input nodes
    layer = arch[0]
    if layer[0] in CONV_LAYERS:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] in (nn.Linear, PositionwiseFeedForward):
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == BasicBlock:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == EquivSetLinear:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == TriplanarGrid:
        triplanar_resolution = layer[1].shape[2]
        in_neuron_idx = torch.arange(3*triplanar_resolution**2)
    else:
        raise ValueError('Invalid first layer')
    
    for i, layer in enumerate(arch):
        out_neuron = (i==len(arch)-1)
        if layer[0] in CONV_LAYERS:
            ret = conv_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] == nn.Linear:
            ret = linear_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] in NORM_LAYERS:
            if layer[0] in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
                norm_type = 'bn'
            elif layer[0] == nn.LayerNorm:
                norm_type = 'ln'
            elif layer[0] == nn.GroupNorm:
                norm_type = 'gn'
            elif layer[0] in (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d):
                norm_type = 'in'
            else:
                raise ValueError('Invalid norm type')
            ret = norm_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops, norm_type=norm_type)
        elif layer[0] == BasicBlock:
            ret = basic_block_to_graph(layer[1:], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == PositionwiseFeedForward:
            ret = ffn_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == SelfAttention:
            ret = self_attention_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 2
        elif layer[0] == EquivSetLinear:
            ret = equiv_set_linear_to_graph(layer[1], layer[2], layer[3], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 1
        elif layer[0] == TriplanarGrid:
            ret = triplanar_to_graph(layer[1], layer_num, out_neuron=out_neuron, curr_idx=curr_idx)
            layer_num += 1
        else:
            raise ValueError('Invalid layer type')
        in_neuron_idx = ret['out_neuron_idx']
            
        edge_index.append(ret['edge_index'])
        edge_attr.append(ret['edge_attr'])
        if ret['added_x'] is not None:
            feat = ret['added_x']
            x.append(feat)
            curr_idx += feat.shape[0]

    x = torch.cat(x, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    edge_attr = torch.cat(edge_attr, dim=0)
    return x, edge_index, edge_attr

def graph_to_arch(arch, weights):
    # arch is the original arch
    arch_new = []
    curr_idx = 0
    for l, layer in enumerate(arch):
        lst = [layer[0]]
        if layer[0] != SelfAttention:
            for tensor in layer[1:]:
                if tensor is not None:
                    weight_size = math.prod(tensor.shape)
                    reshaped = weights[curr_idx:curr_idx+weight_size].reshape(tensor.shape)
                    lst.append(reshaped)
                    curr_idx += weight_size
                else:
                    lst.append(None)
        else:
            # handle in_proj stuff differently, because pytorch stores it all as a big matrix
            in_proj_weight_shape = layer[1].shape
            dim = in_proj_weight_shape[1]
            in_proj_weight = []
            in_proj_bias = []
            for _ in range(3):
                # get q, k, and v
                weight_size = dim*dim
                reshaped = weights[curr_idx:curr_idx+weight_size].reshape(dim, dim)
                in_proj_weight.append(reshaped)
                curr_idx += weight_size
                
                bias_size = dim
                reshaped = weights[curr_idx:curr_idx+bias_size].reshape(dim)
                in_proj_bias.append(reshaped)
                curr_idx += bias_size 
                
            # concatenate q, k, v weights and biases
            lst.append(torch.cat(in_proj_weight, 0))
            lst.append(torch.cat(in_proj_bias, 0))
            
            # out_proj handled normally
            for tensor in layer[3:]:
                if tensor is not None:
                    weight_size = math.prod(tensor.shape)
                    reshaped = weights[curr_idx:curr_idx+weight_size].reshape(tensor.shape)
                    lst.append(reshaped)
                    curr_idx += weight_size
                else:
                    lst.append(None)
        
        # handle residual connections, and other edges that don't correspond to weights
        if layer[0] == PositionwiseFeedForward:
            residual_size = layer[1].shape[1]
            curr_idx += residual_size
        elif layer[0] == BasicBlock:
            residual_size = layer[1].shape[0]
            curr_idx += residual_size
        elif layer[0] == SelfAttention:
            residual_size = layer[1].shape[1]
            curr_idx += residual_size
            
        arch_new.append(lst)
    return arch_new

def arch_to_sequential(arch, model):
    # model is a model of the correct architecture
    arch_idx = 0
    for child in model.children():
        if len(list(child.parameters())) > 0:
            layer = arch[arch_idx]
            sd = child.state_dict()
            layer_idx = 1
            for i, k in enumerate(sd):
                if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                param = nn.Parameter(layer[layer_idx])
                sd[k] = param
                layer_idx += 1
            child.load_state_dict(sd)
            arch_idx += 1
    return model

    
def tests():
    import networkx as nx
    from torch_geometric.utils import to_networkx
    from gmn.graph_construct.net_makers import make_transformer, make_resnet

    def test1(model):
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        
        # number of edges consistent
        assert edge_index.shape[1] == edge_attr.shape[0]
        
        # nodes in edge index exist
        assert x.shape[0] == edge_index.max() + 1
        
        # each node is in some edge
        assert (torch.arange(x.shape[0]) == edge_index.unique()).all()
        
        num_params = sum([p.numel() for p in model.parameters()])
        # at least one edge for each param (not exact because residuals)
        assert edge_index.shape[1] >= num_params
    
    def test2(model):
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        new_arch = graph_to_arch(arch, edge_attr[:, 0])
        new_model = arch_to_sequential(arch, deepcopy(model))
        
        # make sure original objects are reconstructed correctly
        for i in range(len(arch)):
            for j in range(1, len(arch[i])):
                eq = arch[i][j] == new_arch[i][j]
                if type(eq) == torch.Tensor:
                    eq = eq.all()
                assert eq
        
        # check state dicts are the same
        sd1, sd2 = model.state_dict(), new_model.state_dict()
        for k, v in sd1.items():
            assert (v == sd2[k]).all()
            
    def test3(model):
        ''' checks graph properties'''
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        G = to_networkx(data)
        
        # G should be a DAG
        assert nx.is_directed_acyclic_graph(G)
        
        # G should be weakly connected
        assert nx.is_weakly_connected(G)
        
    def test4():
        # hard coded some small neural networks
        
        model = nn.Sequential(BasicBlock(1, 1))
        model(torch.randn(16, 1, 5, 4))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 7
        assert num_params == 22
        assert edge_index.shape[1] == edge_attr.shape[0] == 23
        assert (edge_index == torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 6, 0],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]])).all()
        
        model = nn.Sequential(BasicBlock(1, 2))
        model(torch.randn(16, 1, 5, 4))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 13
        assert num_params == 68
        assert edge_index.shape[1] == edge_attr.shape[0] == 70
        # lmao i actually wrote this all out by hand
        assert (edge_index == torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 7, 8, 8, 0, 0, 11, 11, 12, 12, 9, 10],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 5, 6, 9, 10, 9, 10, 9, 10, 5, 6]]) ).all()
        
        
        model = nn.Sequential(nn.Linear(2, 3, bias=False), nn.ReLU(), nn.LayerNorm(3), nn.Linear(3, 1))
        model(torch.randn(16, 2))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 9
        assert num_params == 16
        assert edge_index.shape[1] == edge_attr.shape[0] == 16
        assert (edge_index == torch.tensor([[0, 0, 0, 1, 1, 1, 5, 5, 5, 6, 6, 6, 2, 3, 4, 8],
                                            [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 7, 7, 7, 7]]) ).all()
        
        model = nn.Sequential(nn.Conv1d(1, 3, 2, bias=False), nn.ReLU(), nn.BatchNorm1d(3), nn.AdaptiveAvgPool1d(1), Flatten(), nn.Linear(3, 2))
        model(torch.randn(16, 1, 5))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 9
        assert num_params == 20
        assert edge_index.shape[1] == edge_attr.shape[0] == 20
        assert (edge_index == torch.tensor([[0, 0, 0, 0, 0, 0, 4, 4, 4, 5, 5, 5, 1, 1, 2, 2, 3, 3, 8, 8],
                                            [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7]]) ).all()
        
        # small attention layer
        model = nn.Sequential(nn.Linear(1,1, bias=False), nn.LayerNorm(1), SelfAttention(1,1))
        model(torch.randn(16, 5, 1))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 10
        assert num_params == 11
        assert edge_index.shape[1] == edge_attr.shape[0] == 12
        assert (edge_index == torch.tensor([[0, 2, 3, 1, 5, 1, 6, 1, 7, 4, 9, 1],
                                            [1, 1, 1, 4, 4, 4, 4, 4, 4, 8, 8, 8]]) ).all()
        
        
        model = make_transformer(1, 1, 1, 1, num_layers=1, vit=False)
        model(torch.randn(16, 8, 1))
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 22
        assert num_params == 29
        assert edge_index.shape[1] == edge_attr.shape[0] == 31
        expected_edge_index = torch.tensor([[0, 2, 3, 4, 1, 6, 1, 7, 1, 8, 5, 10, 1, 11, 12,  9,  9,  9,  9, 17, 17, 17, 17, 13, 14, 15, 16, 19,  9, 18, 21],
                                            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 9,  9, 9,  9,  9, 13, 14, 15, 16, 13, 14, 15, 16, 18, 18, 18, 18, 18, 18, 20, 20]])
        assert( (edge_index == expected_edge_index).all())
        
        model = nn.Sequential(EquivSetLinear(1, 2), nn.GroupNorm(1, 2), nn.ReLU(), EquivSetLinear(2,1))
        assert model(torch.randn(16, 1, 4)).shape == (16, 1, 4)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 8
        assert num_params == 15
        assert edge_index.shape[1] == edge_attr.shape[0] == 15
        assert (edge_index == torch.tensor([ [0, 0, 3, 3, 0, 0, 4, 4, 5, 5, 1, 2, 7, 1, 2],
                                             [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 6, 6, 6, 6, 6] ]) ).all()
        
        model = nn.Sequential(TriplanarGrid(2,1), nn.ReLU(), nn.Linear(4, 1))
        assert model(torch.randn(10, 3)*.1).shape == (10, 1)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 18
        assert num_params == 17
        assert edge_index.shape[1] == edge_attr.shape[0] == 17
        assert (edge_index == torch.tensor([ [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 15, 17],
                                             [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16] ]) ).all()
        
        model = nn.Sequential(TriplanarGrid(2,2), nn.ReLU(), nn.Linear(5, 1))
        assert model(torch.randn(10, 3)*.1).shape == (10, 1)
        arch = sequential_to_arch(model)
        x, edge_index, edge_attr = arch_to_graph(arch)
        num_params = sum(p.numel() for p in model.parameters())
        assert x.shape[0] == 19
        assert num_params == 30
        assert edge_index.shape[1] == edge_attr.shape[0] == 30
        assert (edge_index == torch.tensor([ [3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 0, 1, 2, 15, 16, 18],
                                             [15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 17, 17, 17, 17, 17, 17] ]) ).all()
    
    model1 = nn.Sequential(nn.Linear(2,3), nn.SiLU(), nn.Linear(3,3), nn.ReLU(), nn.Linear(3,2))
    model2 = nn.Sequential(nn.Linear(2,3), nn.LayerNorm(3), nn.SiLU(), nn.Linear(3,3), nn.ReLU(), nn.Linear(3,2))
    model3 = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                      nn.Conv2d(4, 5, 3), nn.ReLU(),
                      nn.AdaptiveAvgPool2d((1,1)), Flatten(),
                      nn.LayerNorm(5), nn.Linear(5, 1))
    model4 = nn.Sequential(PositionwiseFeedForward(10, 20), nn.ReLU(), nn.Linear(10,4))
    model5 = nn.Sequential(nn.Conv2d(2, 3, 3, bias=False), nn.BatchNorm2d(3),
                           nn.ReLU(), nn.Conv2d(3, 3, 3),
                           nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
                           Flatten(), nn.Linear(3,5), nn.ReLU(),
                           nn.Linear(5, 1, bias=False))
    
    model6 = nn.Sequential(nn.Conv2d(3, 2, 3), nn.ReLU(), BasicBlock(2,4), nn.AdaptiveAvgPool2d((1,1)), Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3,1))
    model7 = nn.Sequential(nn.Conv2d(3, 2, 3), nn.ReLU(), BasicBlock(2,2), nn.AdaptiveAvgPool2d((1,1)), Flatten(), nn.Linear(2, 3), nn.ReLU(), nn.Linear(3,1))
    model8 = nn.Sequential(nn.Linear(1,4, bias=False), nn.LayerNorm(4), SelfAttention(4,1), PositionwiseFeedForward(4, 16))
    model9 = make_transformer(3, 16, 4, 2, num_layers=3, vit=True, patch_size=4)
    model10 = make_transformer(3, 16, 4, 2, num_layers=3, vit=False)
    model11 = make_resnet(conv_layers=2, hidden_dim=8, in_dim=3, num_classes=4)
    model12 = nn.Sequential(EquivSetLinear(2,3), nn.ReLU(), EquivSetLinear(3, 1))
    model13 = nn.Sequential(TriplanarGrid(4,2), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1))
    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13]
    
    for i, model in enumerate(models):
        print('Model:', i+1)
        test1(model)
        test2(model)
        test3(model)
        
    test4()
    
    print('Tests pass!')

if __name__ == '__main__':
    tests()

