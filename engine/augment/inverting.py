# -*- coding: utf-8 -*-
"""
@author: 
"""

import copy
import numpy as np


def inverting(data):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    data_aug = copy.deepcopy(data)
    data_aug = np.flip(data_aug, axis=1)

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

