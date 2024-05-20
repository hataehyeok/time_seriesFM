# -*- coding: utf-8 -*-
"""
@author: 
"""


import copy
import numpy as np
import torch
import torch.nn as nn


class MixingUpEncoder(nn.Module):
    def __init__(self, encoder, alpha=1.0):
        r"""
        The MixingUp model described in the paper 'Self-Supervised Representation
        Learning for Time Series '. The implementation is abased on the
        github repository https://github.com/mims-harvard/TFC-pretraining

        Args:
            encoder (Module): The base encoder
            alpha (float, optional): the alpha for beta distribution.
                Default: 1.0.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(MixingUpEncoder, self).__init__()

        self.pretrain_name = 'mixup'
        self.encoder = copy.deepcopy(encoder)
        self.alpha = alpha

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if not is_augment:
            ts_emb = self.encoder.encode(
                ts, normalize=normalize, to_numpy=to_numpy)
            return ts_emb

        alpha = self.alpha

        n_ts = ts.shape[0]
        ts_0 = copy.deepcopy(ts)
        ts_1 = copy.deepcopy(ts)

        order = np.random.permutation(n_ts)
        ts_1 = ts_1[order, :, :]
        lam = np.random.beta(alpha, alpha)

        ts_aug = lam * ts_0 + (1 - lam) * ts_1

        ts_emb_0 = self.encoder.encode(
            ts_0, normalize=normalize, to_numpy=to_numpy)
        ts_emb_1 = self.encoder.encode(
            ts_1, normalize=normalize, to_numpy=to_numpy)
        ts_emb_aug = self.encoder.encode(
            ts_aug, normalize=normalize, to_numpy=to_numpy)

        if to_numpy:
            ts_emb_0 = ts_emb_0.cpu().detach().numpy()
            ts_emb_1 = ts_emb_1.cpu().detach().numpy()
            ts_emb_aug = ts_emb_aug.cpu().detach().numpy()
        return ts_emb_0, ts_emb_1, ts_emb_aug, lam

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

