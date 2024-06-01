# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch.nn as nn
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self, encoder, n_class, n_dim=64, n_layer=2):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.add_module('encoder', encoder)

        in_dim_ = self.encoder.out_dim
        out_dim_ = n_dim
        layers = OrderedDict()
        for i in range(n_layer - 1):
            layers[f'linear_{i:02d}'] = nn.Linear(
                in_dim_, out_dim_)
            layers[f'relu_{i:02d}'] = nn.ReLU()
            in_dim_ = out_dim_
            out_dim_ = n_dim

        layers[f'linear_{n_layer - 1:02d}'] = nn.Linear(
            in_dim_, n_class)
        self.classifier = nn.Sequential(layers)

    def forward(self, ts, normalize=True, to_numpy=False):
        out, hidden = self.encoder.encode(ts, normalize=normalize, to_numpy=False)
        logit = self.classifier(out)
        if to_numpy:
            return logit.cpu().detach().numpy(), hidden.cpu().detach().numpy()
        else:
            return logit, hidden

