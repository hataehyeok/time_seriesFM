# -*- coding: utf-8 -*-
"""
@author: 
"""

import copy
import numpy as np


def shifting(data, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]

    if seed is not None:
        np.random.seed(seed=seed)
    shift_len = np.random.randn(n_data) * data_len
    shift_len = np.round(shift_len)
    shift_len = shift_len.astype(int)

    data_aug = copy.deepcopy(data)
    for i in range(n_data):
        data_aug[i, :] = np.roll(data_aug[i, :], shift_len[i])

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

