# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch


def _normalize_t(t_, normalize):
    if not torch.is_tensor(t_):
        t_ = torch.from_numpy(t_)
    if len(t_.size()) == 1:
        t_ = torch.unsqueeze(t_, 0)
    if len(t_.size()) == 2:
        t_ = torch.unsqueeze(t_, 1)
    if normalize:
        t_mu = torch.mean(t_, 2, keepdims=True)
        t_sigma = torch.std(t_, 2, keepdims=True)
        t_sigma[t_sigma <= 0] = 1.0
        t_ = (t_ - t_mu) / t_sigma
    return t_

