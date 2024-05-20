# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np


def jittering(data, strength=0.1, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    sigma = np.std(data, axis=1, keepdims=True)

    sigma_jitter = strength * sigma
    sigma_jitter[sigma[:, 0] == 0, 0] = strength

    if seed is not None:
        np.random.seed(seed=seed)
    noise = np.random.randn(n_data, data_len) * sigma_jitter
    data_aug = data + noise

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

