# -*- coding: utf-8 -*-
"""
@author: 
"""

import numpy as np


def _cleanup(data, value_min, value_max):
    data = np.round(data)
    data[data < value_min] = value_min
    data[data > value_max] = value_max
    data = data.astype(int)
    return data

