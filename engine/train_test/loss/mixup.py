# -*- coding: utf-8 -*-
"""
@author: 

modify from:
https://github.com/mims-harvard/TFC-pretraining
"""


import torch
from torch import nn


class MixupLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(MixupLoss, self).__init__()
        self.tau = tau

    def forward(self, ts_emb_0, ts_emb_1, ts_emb_aug, lam):
        batch_size = ts_emb_0.size()[0]
        device = ts_emb_0.device

        tau = self.tau

        ts_emb_0 = nn.functional.normalize(ts_emb_0)
        ts_emb_1 = nn.functional.normalize(ts_emb_1)
        ts_emb_aug = nn.functional.normalize(ts_emb_aug)

        labels_lam_0 = lam * torch.eye(batch_size)
        labels_lam_1 = (1 - lam) * torch.eye(batch_size)
        labels = torch.cat((labels_lam_0, labels_lam_1), 1)
        labels = labels.to(device)

        logits = torch.cat((torch.mm(ts_emb_aug, ts_emb_0.T),
                            torch.mm(ts_emb_aug, ts_emb_1.T)), 1)
        loss = _cross_entropy(logits / tau, labels)
        return loss


def _cross_entropy(logits, labels):
    logits = nn.LogSoftmax(dim=1)(logits)
    loss = torch.mean(torch.sum(-labels * logits, 1))
    return loss

