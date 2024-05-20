# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np
from scipy import signal


def time_warping(data, min_ratio=0.5, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]

    if seed is not None:
        np.random.seed(seed=seed)
    ratio = (1 - min_ratio) * np.random.rand(n_data) + min_ratio
    wrap_len = np.round(data_len * ratio)
    wrap_len[wrap_len > data_len] = data_len
    wrap_len = wrap_len.astype(int)

    data_aug = np.zeros((n_data, data_len))
    for i in range(n_data):
        random_vec = np.random.permutation(data_len)
        random_vec = random_vec[:wrap_len[i]]
        random_vec = np.sort(random_vec)
        data_aug_ = data[i, random_vec]
        data_aug[i, :] = signal.resample(data_aug_, data_len)

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

