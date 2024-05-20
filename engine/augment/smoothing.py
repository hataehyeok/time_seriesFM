# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np
from scipy import signal


def smoothing(data, max_ratio=0.5, min_ratio=0.01, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    if seed is not None:
        np.random.seed(seed=seed)
    ratio = np.random.rand(n_data) * (max_ratio - min_ratio) + min_ratio
    data_len_down = np.ceil(data_len * ratio).astype(int)

    data_aug = np.zeros((n_data, data_len))
    for i in range(n_data):
        data_aug_ = signal.resample(data[i, :], data_len_down[i])
        data_aug[i, :] = signal.resample(data_aug_, data_len)

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

