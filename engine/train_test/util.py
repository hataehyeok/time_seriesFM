# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import numpy as np
import torch


def _normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data


def _get_checkpoint(n_ckpt, n_epoch):
    if n_ckpt >= n_epoch:
        ckpts = np.arange(n_epoch)
    else:
        ckpts = np.arange(1, n_ckpt + 1)
        ckpts = n_epoch * (ckpts / n_ckpt) - 1
    ckpts = ckpts.astype(int)
    ckpts_dict = {}
    for ckpt in ckpts:
        ckpts_dict[ckpt] = 0

    last_ckpt = n_epoch - 1
    if last_ckpt not in ckpts_dict:
        ckpts_dict[last_ckpt] = 0
    return ckpts_dict


def _get_start_epoch(model_path, ckpts):
    ckpt_list = [ckpt for ckpt in ckpts] + [0, ]
    ckpt_list = np.array(ckpt_list)
    order = np.sort(-ckpt_list)
    ckpt_list = ckpt_list[order]

    start_epoch = 0
    for i in ckpt_list:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            try:
                pkl = torch.load(model_path_i, map_location='cpu')
            except:
                print(f'{model_path_i} can not be opened. It is removed!')
                os.remove(model_path_i)

        if not os.path.isfile(model_path_i):
            start_epoch = i
    return start_epoch

