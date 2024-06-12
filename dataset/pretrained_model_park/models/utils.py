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

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

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
