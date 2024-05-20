# -*- coding: utf-8 -*-
"""
@author: 
"""

import copy
import numpy as np
from scipy import signal
from .util import _cleanup


def add_step(data, min_ratio=0.1, strength=1, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    sigma = np.std(data, axis=1, keepdims=True)

    sigma_scaling = strength * sigma
    sigma_scaling[sigma[:, 0] == 0, 0] = strength

    if seed is not None:
        np.random.seed(seed=seed)
    segment_len = np.random.rand(n_data)
    segment_len = segment_len * (1 - min_ratio) + min_ratio
    segment_len = segment_len * data_len
    segment_start = np.random.rand(n_data)
    segment_start = segment_start * (data_len - segment_len)
    segment_end = segment_start + segment_len

    segment_start = _cleanup(segment_start, 0, data_len)
    segment_end = _cleanup(segment_end, 0, data_len)

    data_aug = copy.deepcopy(data)
    for i in range(n_data):
        data_aug[i, segment_start[i]:segment_end[i]] += sigma_scaling[i]

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

