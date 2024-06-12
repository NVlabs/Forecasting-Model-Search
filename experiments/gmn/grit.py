'''
https://github.com/LiamMa/GRIT/blob/452e068c6800e0f548c8fd83f14578a4151d8eeb/grit/layer/grit_layer.py

From https://github.com/LiamMa/GRIT, no license
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

import opt_einsum as oe


class GritTransformer(torch.nn.Module):
    '''
    '''

    def __init__(self, hidden_dim, num_layers,
                     use_rrwp=False,
                     num_heads=8,
                     dropout=0.0,
                     attn_dropout=0.2,
                     layer_norm=False,
                     batch_norm=True):
        super().__init__()
        assert not use_rrwp

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(GritTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=True,
                norm_e=True,
                O_e=True,
            ))

    def forward(self, x, edge_index, edge_attr, *args, **kwargs):
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr

class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., 
                 edge_enhance=True,
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.act = nn.ReLU()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, x, edge_index, edge_attr, Q_h, K_h, V_h):
        src = K_h[edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = Q_h[edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        edge_attr = edge_attr.view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = edge_attr[:, :, :self.out_dim], edge_attr[:, :, self.out_dim:]
        # (num relative) x num_heads x out_dim
        score = score * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        attn = score

        # Aggregate with Attn-Score
        msg = V_h[edge_index[0]] * score  # (num relative) x num_heads x out_dim
        wV = torch.zeros_like(V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        if self.edge_enhance:
            rowV = scatter(e_t * score, edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            wV = wV + rowV

        return wV, wE

    def forward(self, x, edge_index, edge_attr):
        Q_h = self.Q(x)
        K_h = self.K(x)

        V_h = self.V(x)
        edge_attr = self.E(edge_attr)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        wV, wE = self.propagate_attention(x, edge_index, edge_attr, Q_h, K_h, V_h)
        h_out = wV
        e_out = wE

        return h_out, e_out

class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 update_e=True,
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = update_e
        self.bn_momentum = .1
        self.bn_no_runner = False
        self.rezero = False

        self.act = nn.ReLU()
        self.deg_scaler = True

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=False,
            dropout=attn_dropout,
            clamp=5.,
            edge_enhance=True,
        )

        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, x, edge_index, edge_attr):
        h = x
        #num_nodes = batch.num_nodes
        num_nodes = x.shape[0]
        log_deg = get_log_deg(num_nodes, edge_index)

        h_in1 = h  # for first residual connection
        e_in1 = edge_attr
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(h, edge_index, edge_attr)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        x = h
        if self.update_e:
            edge_attr = e
        else:
            edge_attr = e_in1

        return x, edge_attr

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )

@torch.no_grad()
def get_log_deg(num_nodes, edge_index):
    deg = torch_geometric.utils.degree(edge_index[1],
                           num_nodes=num_nodes,
                           dtype=torch.float
                           )
    log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(num_nodes, 1)
    return log_deg

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
