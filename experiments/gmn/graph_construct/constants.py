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

import torch.nn as nn

from gmn.graph_construct.layers import PositionwiseFeedForward

CONV_LAYERS = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
NORM_LAYERS = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.LayerNorm,
               nn.GroupNorm,
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]

NODE_TYPES = {'neuron': 0,
              'bias': 1,
              'channel': 2,
              'channel_bias': 3,
              'bn_gamma': 4,
              'bn_beta': 5,
              'ln_gamma': 6,
              'ln_beta': 7,
              'gn_gamma': 8,
              'gn_beta': 9,
              'in_gamma': 10,
              'in_beta': 11,
              'attention_neuron': 12,
              'attention_bias': 13,
              'deepsets_neuron': 14,
              'deepsets_bias': 15,
              'triplanar': 16}

EDGE_TYPES = {'lin_weight': 0,
              'lin_bias': 1,
              'conv_weight': 2,
              'conv_bias': 3,
              'residual': 4,
              'bn_gamma': 5,
              'bn_beta': 6,
              'ln_gamma': 7,
              'ln_beta': 8,
              'gn_gamma': 9,
              'gn_beta': 10,
              'in_gamma': 11,
              'in_beta': 12,
              'triplanar': 13}
RESIDUAL_LAYERS = {PositionwiseFeedForward}
