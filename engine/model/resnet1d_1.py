# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
import torch.nn as nn
from collections import OrderedDict
from .util import _normalize_t


class ResNet1D(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, n_dim=64,
                 block_type='standard', norm=None,
                 is_projector=True, project_norm=None):
        r"""
        1D ResNet-based time series encoder

        Args:
            in_dim (int, optional): Number of dimension for the input time
                series. Default: 1.
            out_dim (int, optional): Number of dimension for the output
                representation. Default: 128.
            n_dim (int, optional): Number of base dimension for the
                intermediate representation. Default: 64.
            block_type (string, optional): If set to ``standard``, the encoder
                will use the standard residual block for 1D ResNet. If set to
                ``alternative``, the encoder will use the alternative residual
                block inspired by the paper 'On Layer Normalization in the
                Transformer Architecture'. Default: ``standard``. See 'Deep
                learning for time series classification: a review' for the
                details.
            norm (string, optional): If set to ``BN``, the encoder will
                use batch normalization. If set to ``LN``, the encoder will
                use layer normalization. If set to None, the encoder will
                not use normalization. Default: None (no normalization).
            is_projector (bool, optional): If set to ``False``, the encoder
                will not use additional projection layers. Default: ``True``.
            project_norm (string, optional): If set to ``BN``, the projector
                will use batch normalization. If set to ``LN``, the projector
                will use layer normalization. If set to None, the projector
                will not use normalization. Default: None (no normalization).

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`, :math:`(N, L_{in})`, or
                :math:`(L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(ResNet1D, self).__init__()
        assert block_type in ['standard', 'alternative', ]
        assert norm in ['BN', 'LN', None]
        assert project_norm in ['BN', 'LN', None]

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dim = n_dim
        self.is_projector = is_projector

        if block_type == 'standard':
            Block = Block_Standard
        elif block_type == 'alternative':
            Block = Block_Alt

        self.in_net = nn.Conv1d(
            in_dim, n_dim, 7, stride=2, padding=3, dilation=1)
        self.add_module('in_net', self.in_net)
        res_net_layer = OrderedDict()
        res_net_layer['block_0'] = Block(n_dim, n_dim, norm)
        res_net_layer['block_1'] = Block(n_dim, n_dim * 2, norm)
        res_net_layer['block_2'] = Block(n_dim * 2, n_dim * 2, norm)
        res_net_layer['pooling'] = nn.AdaptiveAvgPool1d(1)
        self.res_net_layer = res_net_layer
        self.res_net = nn.Sequential(res_net_layer)

        self.out_net = nn.Linear(n_dim * 2, out_dim)
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

    def forward(self, ts, normalize=True, to_numpy=False):
        device = self.dummy.device
        is_projector = self.is_projector

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        ts_emb = self.res_net(ts_emb)
        ts_emb = ts_emb[:, :, 0]
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            ts_emb = self.projector(ts_emb)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False):
        return self.forward(ts, normalize=normalize, to_numpy=to_numpy)

    def encode_seq(self, ts, normalize=True, to_numpy=False):
        device = self.dummy.device
        is_projector = self.is_projector

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        ts_emb = self.res_net_layer['block_0'](ts_emb)
        ts_emb = self.res_net_layer['block_1'](ts_emb)
        ts_emb = self.res_net_layer['block_2'](ts_emb)
        ts_emb = torch.transpose(ts_emb, 1, 2)
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            project_norm = self.project_norm
            if project_norm == 'BN':
                layers = [module for module in projector.modules()]
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
        ts_emb = torch.transpose(ts_emb, 1, 2)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb


class Block_Standard(nn.Module):
    def __init__(self, in_dim, n_dim, norm):
        super(Block_Standard, self).__init__()

        main_pass = OrderedDict()
        main_pass['cov_0'] = nn.Conv1d(
            in_dim, n_dim, 7, stride=1, padding=3, dilation=1)
        if norm == 'BN':
            main_pass['bn_0'] = nn.BatchNorm1d(n_dim)
        elif norm == 'LN':
            main_pass['ln_0'] = LayerNormT(n_dim)
        main_pass['relu_0'] = nn.ReLU()

        main_pass['cov_1'] = nn.Conv1d(
            n_dim, n_dim, 5, stride=1, padding=2, dilation=1)
        if norm == 'BN':
            main_pass['bn_1'] = nn.BatchNorm1d(n_dim)
        elif norm == 'LN':
            main_pass['ln_1'] = LayerNormT(n_dim)
        main_pass['relu_1'] = nn.ReLU()

        main_pass['cov_2'] = nn.Conv1d(
            n_dim, n_dim, 3, stride=1, padding=1, dilation=1)
        if norm == 'BN':
            main_pass['bn_2'] = nn.BatchNorm1d(n_dim)
        elif norm == 'LN':
            main_pass['ln_2'] = LayerNormT(n_dim)
        self.main_pass = nn.Sequential(main_pass)

        shortcut = OrderedDict()
        if in_dim != n_dim:
            shortcut['cov_0'] = nn.Conv1d(
                in_dim, n_dim, 1, stride=1, padding=0, dilation=1)
            if norm == 'BN':
                shortcut['bn_0'] = nn.BatchNorm1d(n_dim)
            elif norm == 'LN':
                main_pass['ln_0'] = LayerNormT(n_dim)
        else:
            shortcut['id_0'] = nn.Identity()
        self.shortcut = nn.Sequential(shortcut)

    def forward(self, data):
        hodden_0 = self.main_pass(data)
        hodden_1 = self.shortcut(data)
        output = nn.ReLU()(hodden_0 + hodden_1)
        return output


class Block_Alt(nn.Module):
    def __init__(self, in_dim, n_dim, norm):
        super(Block_Alt, self).__init__()

        main_pass = OrderedDict()
        if norm == 'BN':
            main_pass['bn_0'] = nn.BatchNorm1d(in_dim)
        elif norm == 'LN':
            main_pass['ln_0'] = LayerNormT(in_dim)
        main_pass['cov_0'] = nn.Conv1d(
            in_dim, n_dim, 7, stride=1, padding=3, dilation=1)
        main_pass['relu_0'] = nn.ReLU()

        if norm == 'BN':
            main_pass['bn_1'] = nn.BatchNorm1d(n_dim)
        elif norm == 'LN':
            main_pass['ln_1'] = LayerNormT(n_dim)
        main_pass['cov_1'] = nn.Conv1d(
            n_dim, n_dim, 5, stride=1, padding=2, dilation=1)
        main_pass['relu_1'] = nn.ReLU()

        if norm == 'BN':
            main_pass['bn_2'] = nn.BatchNorm1d(n_dim)
        elif norm == 'LN':
            main_pass['ln_2'] = LayerNormT(n_dim)
        main_pass['cov_2'] = nn.Conv1d(
            n_dim, n_dim, 3, stride=1, padding=1, dilation=1)
        main_pass['relu_2'] = nn.ReLU()
        self.main_pass = nn.Sequential(main_pass)

        shortcut = OrderedDict()
        if in_dim != n_dim:
            if norm == 'BN':
                shortcut['bn_0'] = nn.BatchNorm1d(in_dim)
            elif norm == 'LN':
                main_pass['ln_0'] = LayerNormT(in_dim)
            shortcut['cov_0'] = nn.Conv1d(
                in_dim, n_dim, 1, stride=1, padding=0, dilation=1)
        else:
            shortcut['id_0'] = nn.Identity()
        self.shortcut = nn.Sequential(shortcut)

    def forward(self, data):
        hodden_0 = self.main_pass(data)
        hodden_1 = self.shortcut(data)
        output = hodden_0 + hodden_1
        return output


class LayerNormT(nn.Module):
    def __init__(self, n_dim):
        super(LayerNormT, self).__init__()
        self.layer_norm = nn.LayerNorm(n_dim)
        self.add_module('layer_norm', self.layer_norm)

    def forward(self, data):
        data = torch.transpose(data, 1, 2)
        self.layer_norm.forward(data)
        data = torch.transpose(data, 1, 2)
        return data

