# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


def _apply_warping(data, n_knot, strength, seed):
    np.random.seed(seed=seed)
    data_len = data.shape[0]
    knot_step = data_len / (n_knot - 1)
    # knot_t = np.arange(knot_step, data_len, knot_step)
    knot_t = np.arange(0, data_len + knot_step, knot_step)
    knot_mag = np.random.randn(n_knot) * strength + 1
    data_aug_t = np.arange(data_len)

    if knot_t.shape[0] != knot_mag.shape[0]:
        knot_t = knot_t[:knot_mag.shape[0]]
    # data_aug = data + CubicSpline(knot_t, knot_mag)(data_aug_t)
    data_aug = data * CubicSpline(knot_t, knot_mag)(data_aug_t)
    return data_aug


def mag_warping(data, strength=1, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    sigma = np.std(data, axis=1, keepdims=False)

    sigma_scaling = strength * sigma
    sigma_scaling[sigma == 0] = strength

    if seed is not None:
        np.random.seed(seed=seed)
    n_knot = np.random.randint(3, high=data_len, size=n_data)
    seeds = np.random.randint(2 ** 32 - 1, size=n_data)

    data_aug = np.zeros((n_data, data_len))
    for i in range(n_data):
        data_aug[i, :] = _apply_warping(
            data[i, :], n_knot[i], sigma_scaling[i], seeds[i])

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

