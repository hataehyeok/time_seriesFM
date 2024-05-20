# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
import torch.nn as nn
from .util import _normalize_t


class RNNet(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, rnn_type='GRU',
                 n_layer=2, n_dim=64, is_projector=True,
                 project_norm=None, dropout=0.0):
        r"""
        RNN-based time series encoder

        Args:
            in_dim (int, optional): Number of dimension for the input time
                series. Default: 1.
            out_dim (int, optional): Number of dimension for the output
                representation. Default: 128.
            rnn_type (string, optional): The type of RNN cell to use. Can be
                either ``'GRU'`` or ``'LSTM'``. Default: ``'GRU'``
            n_layer (int, optional): Number of layer for the transformer
                encoder. Default: 8.
            n_dim (int, optional): Number of dimension for the intermediate
                representation. Default: 64.
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
        super(RNNet, self).__init__()
        assert project_norm in ['BN', 'LN', None]

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dim = n_dim
        self.is_projector = is_projector

        self.in_net = nn.Conv1d(
            in_dim, n_dim, 7, stride=2, padding=3, dilation=1)
        self.add_module('in_net', self.in_net)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=n_dim, hidden_size=n_dim, num_layers=n_layer,
                batch_first=True, dropout=dropout, bidirectional=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=n_dim, hidden_size=n_dim, num_layers=n_layer,
                batch_first=True, dropout=dropout, bidirectional=True)

        self.out_net = nn.Linear(n_dim * 2, out_dim)
        self.project_norm = project_norm
        if is_projector:
            if project_norm == 'BN':
                self.is_projector = nn.Sequential(
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
            elif project_norm == 'LN':
                self.is_projector = nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(out_dim),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_dim * 2),
                    nn.Linear(out_dim * 2, out_dim)
                )
            else:
                self.is_projector = nn.Sequential(
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
        ts_emb = torch.transpose(ts_emb, 1, 2)
        ts_emb, _ = self.rnn(ts_emb)
        ts_emb = torch.transpose(ts_emb, 1, 2)
        ts_emb = ts_emb[:, :, 0]
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            ts_emb = self.is_projector(ts_emb)

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
        ts_emb = torch.transpose(ts_emb, 1, 2)
        ts_emb, _ = self.rnn(ts_emb)
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
                ts_emb = self.is_projector(ts_emb)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

