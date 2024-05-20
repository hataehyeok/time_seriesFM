# -*- coding: utf-8 -*-
"""
@author: 
"""


import copy
import numpy as np
import torch
import torch.nn as nn


class TS2VecEncoder(nn.Module):
    def __init__(self, encoder):
        r"""
        The TS2Vec model described in the paper 'TS2Vec: Towards Universal
        Representation of Time Series'. The implementation is abased on the
        github repository https://github.com/mims-harvard/TFC-pretraining

        Args:
            encoder (Module): The base encoder

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(TS2VecEncoder, self).__init__()

        self.pretrain_name = 'ts2vec'
        self.encoder = copy.deepcopy(encoder)

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if not is_augment:
            ts_emb = self.encoder.encode_seq(
                ts, normalize=normalize, to_numpy=False)
            ts_emb = nn.AdaptiveMaxPool1d(1)(ts_emb)
            ts_emb = ts_emb[:, :, 0]
            if to_numpy:
                ts_emb = ts_emb.cpu().detach().numpy()
            return ts_emb

        n_ts = ts.shape[0]
        ts_len = ts.shape[2]
        corp_len = np.random.randint(low=4, high=ts_len - 3)
        crop_r_start = np.random.randint(
            low=4, high=ts_len - corp_len + 1,
            size=n_ts)

        low_val = crop_r_start - corp_len + 1
        low_val[low_val < 0] = 0
        crop_l_start = np.random.randint(
            low=low_val, high=crop_r_start,
            size=n_ts)

        corp_len = int(corp_len)
        crop_l_start = crop_l_start.astype(int)
        crop_r_start = crop_r_start.astype(int)

        ts_l = _get_corp(ts, crop_l_start, corp_len)
        ts_r = _get_corp(ts, crop_r_start, corp_len)
        ts_emb_l = self.encoder.encode_seq(
            ts_l, normalize=normalize, to_numpy=False)
        ts_emb_r = self.encoder.encode_seq(
            ts_r, normalize=normalize, to_numpy=False)
        if to_numpy:
            ts_emb_l = ts_emb_l.cpu().detach().numpy()
            ts_emb_r = ts_emb_r.cpu().detach().numpy()
        return ts_emb_l, ts_emb_r

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.forward(
            ts, normalize=normalize, to_numpy=to_numpy, is_augment=False)
        return ts_emb


def _get_corp(ts, corp_start, corp_len):
    n_ts = ts.shape[0]
    n_dim = ts.shape[1]
    corp = np.zeros((n_ts, n_dim, corp_len))
    for i in range(n_ts):
        corp[i, :, :] = ts[i, :, corp_start[i]:corp_start[i] + corp_len]
    return corp

