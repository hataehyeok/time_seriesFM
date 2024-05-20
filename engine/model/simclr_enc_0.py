# -*- coding: utf-8 -*-
"""
@author: 
"""


import copy
import numpy as np
import torch
import torch.nn as nn


class SimCLREncoder(nn.Module):
    def __init__(self, encoder):
        r"""
        The SimCLR model described in the paper 'Exploring Contrastive
        Learning in Human Activity Recognition for Healthcare'. The
        implementation is abased on the github repository
        https://github.com/mims-harvard/TFC-pretraining

        Args:
            encoder (Module): The base encoder

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(SimCLREncoder, self).__init__()

        self.pretrain_name = 'simclr'
        self.encoder = copy.deepcopy(encoder)

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if not is_augment:
            ts_emb = self.encoder.encode(
                ts, normalize=normalize, to_numpy=to_numpy)
            return ts_emb

        ts_aug = _augment_ts(ts)
        ts_emb_aug = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb_aug

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb


def _augment_ts(ts):
    ts_aug = copy.deepcopy(ts)
    ts_aug = _scaling_transform_vectorized(ts_aug)
    ts_aug = _negate_transform_vectorized(ts_aug)
    return ts_aug


def _scaling_transform_vectorized(X, sigma=0.1):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(
        loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor


def _negate_transform_vectorized(X):
    """
    Inverting the signals
    """
    return X * -1

