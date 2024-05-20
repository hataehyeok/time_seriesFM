# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np


def flipping(data):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    mu = np.mean(data, axis=1, keepdims=True)

    data_aug = 2 * mu - data

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

