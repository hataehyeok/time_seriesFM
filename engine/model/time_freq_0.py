# -*- coding: utf-8 -*-
"""
@author: 
"""


import copy
import numpy as np
import torch
import torch.nn as nn
from ..augment import jittering


class TimeFreqEncoder(nn.Module):
    def __init__(self, encoder,
                 jitter_strength=0.1, freq_ratio=0.1,
                 freq_strength=0.1, project_norm=None):
        r"""
        The TF-C model described in the paper 'Self-Supervised Contrastive
        Pre-Training For Time Series via Time-Frequency Consistency'. The
        implementation is abased on the authors' github repository
        https://github.com/mims-harvard/TFC-pretraining

        Args:
            encoder (Module): The base encoder for both time/frequency-based
                contrastive learning
            jitter_strength (float, optional): the strength of jitter added
                when creating augmented time series in time domain.
            freq_ratio (float, optional): ratio of perturbed frequencies for
                frequency removal/amplification when creating augmented time
                series in frequency domain.
            freq_strength (float, optional): strength of frequency
                amplification  when creating augmented time series in
                frequency domain.
            project_norm (string, optional): If set to ``BN``, the projector
                will use batch normalization. If set to ``LN``, the projector
                will use layer normalization. If set to None, the projector
                will not use normalization. Default: None (no normalization).

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(TimeFreqEncoder, self).__init__()
        assert project_norm in ['BN', 'LN', None]

        self.pretrain_name = 'timefreq'
        self.encoder_t = copy.deepcopy(encoder)
        self.encoder_f = copy.deepcopy(encoder)
        self.add_module('encoder_t', self.encoder_t)
        self.add_module('encoder_f', self.encoder_f)

        self.jitter_strength = jitter_strength
        self.freq_ratio = freq_ratio
        self.freq_strength = freq_strength

        out_dim_t = self.encoder_t.out_dim
        out_dim_f = self.encoder_f.out_dim
        self.out_dim = out_dim_t + out_dim_f

        if project_norm == 'BN':
            self.projector_t = nn.Sequential(
                nn.BatchNorm1d(out_dim_t),
                nn.ReLU(),
                nn.Linear(out_dim_t, out_dim_t * 2),
                nn.BatchNorm1d(out_dim_t * 2),
                nn.ReLU(),
                nn.Linear(out_dim_t * 2, out_dim_t)
            )
        elif project_norm == 'LN':
            self.projector_t = nn.Sequential(
                nn.ReLU(),
                nn.LayerNorm(out_dim_t),
                nn.Linear(out_dim_t, out_dim_t * 2),
                nn.ReLU(),
                nn.LayerNorm(out_dim_t * 2),
                nn.Linear(out_dim_t * 2, out_dim_t)
            )
        else:
            self.projector_t = nn.Sequential(
                nn.ReLU(),
                nn.Linear(out_dim_t, out_dim_t * 2),
                nn.ReLU(),
                nn.Linear(out_dim_t * 2, out_dim_t)
            )
        self.add_module('projector_t', self.projector_t)

        if project_norm == 'BN':
            self.projector_f = nn.Sequential(
                nn.BatchNorm1d(out_dim_f),
                nn.ReLU(),
                nn.Linear(out_dim_f, out_dim_f * 2),
                nn.BatchNorm1d(out_dim_f * 2),
                nn.ReLU(),
                nn.Linear(out_dim_f * 2, out_dim_f)
            )
        elif project_norm == 'LN':
            self.projector_f = nn.Sequential(
                nn.ReLU(),
                nn.LayerNorm(out_dim_f),
                nn.Linear(out_dim_f, out_dim_f * 2),
                nn.ReLU(),
                nn.LayerNorm(out_dim_f * 2),
                nn.Linear(out_dim_f * 2, out_dim_f)
            )
        else:
            self.projector_f = nn.Sequential(
                nn.ReLU(),
                nn.Linear(out_dim_f, out_dim_f * 2),
                nn.ReLU(),
                nn.Linear(out_dim_f * 2, out_dim_f)
            )
        self.add_module('projector_f', self.projector_f)
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        ts_t = _normalize_t(ts, normalize)
        n_dim = ts_t.shape[1]
        ts_f = np.fft.fft(ts_t, axis=2)
        ts_f = np.abs(ts_f)

        if is_augment:
            jitter_strength = self.jitter_strength
            for i in range(n_dim):
                ts_t[:, i, :] = jittering(
                    ts_t[:, i, :], strength=jitter_strength,
                    seed=None)

            freq_ratio = self.freq_ratio
            freq_strength = self.freq_strength
            for i in range(n_dim):
                ts_f[:, i, :] = _freq_perturb(
                    ts_f[:, i, :], ratio=freq_ratio, strength=freq_strength,
                    seed=None)

        h_t = self.encoder_t.encode(
            ts_t, normalize=False, to_numpy=False)
        z_t = self.projector_t(h_t)

        h_f = self.encoder_f.encode(
            ts_f, normalize=False, to_numpy=False)
        z_f = self.projector_f(h_f)

        if to_numpy:
            h_t = h_t.cpu().detach().numpy()
            z_t = z_t.cpu().detach().numpy()
            h_f = h_f.cpu().detach().numpy()
            z_f = z_f.cpu().detach().numpy()
        return h_t, z_t, h_f, z_f

    def encode(self, ts, normalize=True, to_numpy=False):
        _, z_t, _, z_f = self.forward(
            ts, normalize=normalize, to_numpy=False, is_augment=False)
        feature = torch.cat((z_t, z_f), dim=1)
        if to_numpy:
            return feature.cpu().detach().numpy()
        else:
            return feature


def _freq_perturb(data, ratio=0.1, strength=0.1, seed=None):
    n_data = data.shape[0]
    data_len = data.shape[1]

    data_aug = copy.deepcopy(data)
    if seed is not None:
        np.random.seed(seed=seed)
    if ratio < 1:
        mask_remove = np.random.rand(n_data, data_len)
        mask_remove = mask_remove < ratio
        data_aug[mask_remove] = 0.0

        mask_perturb = np.random.rand(n_data, data_len)
        mask_perturb = mask_perturb < ratio

    sigma = np.std(data, axis=1, keepdims=True)
    sigma_scaling = strength * sigma
    sigma_scaling[sigma == 0] = strength

    noise = np.random.rand(n_data, data_len) * sigma_scaling
    if ratio < 1:
        data_aug[mask_perturb] = data_aug[mask_perturb] + noise[mask_perturb]
    else:
        data_aug = data_aug + noise
    return data_aug


def _normalize_t(t_, normalize):
    if normalize:
        t_mu = np.mean(t_, axis=2, keepdims=True)
        t_sigma = np.std(t_, axis=2, keepdims=True)
        t_sigma[t_sigma <= 0] = 1
        t_ = (t_ - t_mu) / t_sigma
    return t_

