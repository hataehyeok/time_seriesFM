# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np


def scaling(data, strength=1, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    sigma = np.std(data, axis=1, keepdims=True)
    mu = np.mean(data, axis=1, keepdims=True)

    sigma_scaling = strength * sigma
    sigma_scaling[sigma[:, 0] == 0, 0] = strength

    if seed is not None:
        np.random.seed(seed=seed)
    noise = np.random.randn(n_data, 1) * sigma_scaling + 1
    data_aug = (data - mu) * noise + mu

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

