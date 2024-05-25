# -*- coding: utf-8 -*-
"""
@author: 
"""


import math
import torch
import torch.nn as nn
from collections import OrderedDict
from .util import _normalize_t


class Transformer(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, n_layer=8, n_dim=64, n_head=8,
                 norm_first=False, is_pos=True, is_projector=True,
                 project_norm=None, dropout=0.0, aggregation_mode='class_token', pooling_mode='gt'):
        r"""
        Transformer-based time series encoder

        Args:
            in_dim (int, optional): Number of dimension for the input time
                series. Default: 1.
            out_dim (int, optional): Number of dimension for the output
                representation. Default: 128.
            n_layer (int, optional): Number of layer for the transformer
                encoder. Default: 8.
            n_dim (int, optional): Number of dimension for the intermediate
                representation. Default: 64.
            n_head (int, optional): Number of head for the transformer
                encoder. Default: 8.
            norm_first: if ``True``, layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise it's done
                after. Default: ``False`` (after).
            is_pos (bool, optional): If set to ``False``, the encoder will
                not use position encoding. Default: ``True``.
            is_projector (bool, optional): If set to ``False``, the encoder
                will not use additional projection layers. Default: ``True``.
            project_norm (string, optional): If set to ``BN``, the projector
                will use batch normalization. If set to ``LN``, the projector
                will use layer normalization. If set to None, the projector
                will not use normalization. Default: None (no normalization).
            dropout (float, optional): The probability of an element to be
                zeroed for the dropout layers. Default: 0.0.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`, :math:`(N, L_{in})`, or
                :math:`(L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(Transformer, self).__init__()
        assert project_norm in ['BN', 'LN', None]

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dim = n_dim
        self.is_projector = is_projector
        self.is_pos = is_pos
        self.max_len = 0
        self.dropout = dropout
        self.num_segments = 8
        self.aggregation_mode = aggregation_mode
        self.pooling_mode = pooling_mode

        self.in_net = nn.Conv1d(
            in_dim, n_dim, 7, stride=2, padding=3, dilation=1)
        self.add_module('in_net', self.in_net)
        transformer = OrderedDict()
        for i in range(n_layer):
            transformer[f'encoder_{i:02d}'] = nn.TransformerEncoderLayer(
                n_dim, n_head, dim_feedforward=n_dim,
                dropout=dropout, batch_first=True,
                norm_first=norm_first)
        self.transformer = nn.Sequential(transformer)

        # Class token initialization
        self.start_token = nn.Parameter(
            torch.randn(1, n_dim, 1))
        self.register_parameter(
            name='start_token',
            param=self.start_token)

        self.linear_st = nn.Linear(self.num_segments * n_dim, n_dim)
        self.linear = nn.Linear(16448, n_dim)
        self.out_net = nn.Linear(n_dim, out_dim)
        self.project_norm = project_norm
        if is_projector:
            if project_norm == 'BN':
                self.projector = nn.Sequential(
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
            elif project_norm == 'LN':
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(out_dim),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_dim * 2),
                    nn.Linear(out_dim * 2, out_dim)
                )
            else:
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
        self.dummy = nn.Parameter(torch.empty(0))

    # Freeze pre-trained backbone
    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.out_net.parameters():
            param.requires_grad = True
        if self.is_projector:
            for param in self.projector.parameters():
                param.requires_grad = True
        self.start_token.requires_grad = True

    # Global temporal pooling
    def gtpool(self, h, op):
        if op == 'avg':
            return torch.mean(h, dim=1)
        if op == 'sum':
            return torch.sum(h, dim=1)
        elif op == 'max':
            return torch.max(h, dim=1)[0]

    # Static temporal pooling
    def stpool(self, h, op):
        segment_sizes = [int(h.shape[1]/self.num_segments)] * self.num_segments
        segment_sizes[-1] += h.shape[1] - sum(segment_sizes)

        hs = torch.split(h, segment_sizes, dim=1)
        if op == 'avg':
            hs = [h_.mean(dim=1, keepdim=True) for h_ in hs]
        if op == 'sum':
            hs = [h_.sum(dim=1, keepdim=True) for h_ in hs]
        elif op == 'max':    
            hs = [h_.max(dim=1)[0].unsqueeze(dim=1) for h_ in hs]
        hs = torch.cat(hs, dim=1)
        return hs

    # Dynamic temporal pooling
    # todo: proto 학습하고 가져와야함
    def dtpool(self, h, op):
        A = self.softdtw.align(self.protos.repeat(h.shape[0], 1, 1), h)
        
        if op == 'avg':
            A /= A.sum(dim=2, keepdim=True)
            h = torch.bmm(h, A.transpose(1, 2))
        elif op == 'sum':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.sum(dim=3)
        elif op == 'max':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.max(dim=3)[0]
        return h


    def forward(self, ts, normalize=True, to_numpy=False, pool_op='avg'):
        device = self.dummy.device
        is_projector = self.is_projector
        is_pos = self.is_pos

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        if is_pos:
            n_dim = self.n_dim
            dropout = self.dropout
            ts_len = ts_emb.size()[2]
            if ts_len > self.max_len:
                self.max_len = ts_len
                self.pos_net = PositionalEncoding(
                    n_dim, ts_len, dropout=dropout)
                self.pos_net.to(device)
            ts_emb = self.pos_net(ts_emb)

        # Add class token to in front of embedding sequence
        start_tokens = self.start_token.expand(ts_emb.size()[0], -1, -1)
        ts_emb = torch.cat((start_tokens, ts_emb, ), dim=2)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        ts_emb = self.transformer(ts_emb)
        # import pdb;pdb.set_trace()
        
        # Previous  : backbone -> flatten -> linear projection
        # Present   : backbone -> pooling -> flatten -> linear projection

        if (self.aggregation_mode == 'class_token'):
            ts_emb = ts_emb[:, 0, :]      # ts_emb = ts_emb.view(ts_emb.size(0), -1)
        elif (self.aggregation_mode == 'flatten'):
            ts_emb = ts_emb.reshape(ts_emb.size(0), -1)
            ts_emb = self.linear(ts_emb)
        elif (self.aggregation_mode == 'pooling'):
            if self.pooling_mode == 'gt':
                ts_emb = self.gtpool(ts_emb, pool_op)
            elif self.pooling_mode == 'st':
                ts_emb = self.stpool(ts_emb, pool_op)
                ts_emb = ts_emb.reshape(ts_emb.size(0), -1)
                ts_emb = self.linear_st(ts_emb)
            elif self.pooling_mode == 'dt':
                ts_emb = self.dtpool(ts_emb, pool_op)
                ts_emb = ts_emb.reshape(ts_emb.size(0), -1)
            else:
                raise ValueError(f"Unsupported pool_type: {self.pooling_mode}")
        
        # Linear projection
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            ts_emb = self.projector(ts_emb)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False, pool_op='avg', pool_type='gt'):
        return self.forward(ts, normalize=normalize, to_numpy=to_numpy)

    def encode_seq(self, ts, normalize=True, to_numpy=False, pool_op='avg', pool_type='gt'):
        device = self.dummy.device
        is_projector = self.is_projector
        is_pos = self.is_pos

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        if is_pos:
            n_dim = self.n_dim
            dropout = self.dropout
            ts_len = ts_emb.size()[2]
            if ts_len > self.max_len:
                self.max_len = ts_len
                self.pos_net = PositionalEncoding(
                    n_dim, ts_len, dropout=dropout)
                self.pos_net.to(device)
            ts_emb = self.pos_net(ts_emb)

        start_tokens = self.start_token.expand(ts_emb.size()[0], -1, -1)
        ts_emb = torch.cat((start_tokens, ts_emb, ), dim=2)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        ts_emb = self.transformer(ts_emb)
        ts_emb = self.out_net(ts_emb)
        if is_projector:
            project_norm = self.project_norm
            if project_norm == 'BN':
                layers = [module for module in is_projector.modules()]
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[1](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[2](ts_emb)
                ts_emb = layers[3](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[4](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[5](ts_emb)
                ts_emb = layers[6](ts_emb)
            else:
                ts_emb = self.projector(ts_emb)

        ts_emb = ts_emb[:, 1:, :]
        start_tokens = ts_emb[:, 0:1, :]
        start_tokens = start_tokens.expand(-1, ts_emb.size()[1], -1)
        ts_emb = ts_emb + start_tokens
        ts_emb = torch.transpose(ts_emb, 1, 2)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, max_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2) * (-math.log(10000.0) / n_dim))
        pos_emb = torch.zeros(1, n_dim, max_len)

        position = position.unsqueeze(0)
        div_term = div_term.unsqueeze(1)
        pos_emb[0, 0::2, :] = torch.sin(div_term * position)
        pos_emb[0, 1::2, :] = torch.cos(div_term * position)
        self.register_buffer('pos_emb', pos_emb, persistent=False)

    def forward(self, x):
        x = x + self.pos_emb[:, :, :x.size()[2]]
        return self.dropout(x)

