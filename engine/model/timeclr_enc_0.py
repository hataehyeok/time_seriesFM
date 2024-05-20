# -*- coding: utf-8 -*-
"""
@author: 
"""


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeCLREncoder(nn.Module):
    def __init__(self, encoder, aug_bank):
        r"""
        The proposed TimeCLR method

        Args:
            encoder (Module): The base encoder
            aug_bank (list): A list of augmentation methods.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(TimeCLREncoder, self).__init__()

        self.pretrain_name = 'timeclr'
        self.encoder = copy.deepcopy(encoder)

        self.aug_bank = aug_bank
        n_aug = len(aug_bank)
        self.n_aug = n_aug

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if is_augment:
            ts = self._augment_ts(ts)

        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

    def _augment_ts(self, ts):
        n_ts = ts.shape[0]
        n_aug = self.n_aug
        ts_aug = copy.deepcopy(ts)
        aug_bank = self.aug_bank
        for i in range(n_ts):
            aug_idx = np.random.randint(n_aug)
            ts_aug[i, 0, :] = aug_bank[aug_idx](ts_aug[i, 0, :])
        return ts_aug

