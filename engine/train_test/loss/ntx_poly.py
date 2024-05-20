# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
import torch
import torch.nn as nn


def _dot_similarity(x):
    return torch.mm(x, x.T)


def _cosine_similarity(x):
    return torch.nn.CosineSimilarity(dim=-1)(
        x.unsqueeze(1), x.unsqueeze(0))


def _get_mask(batch_size, device):
    diag_0 = np.eye(2 * batch_size)
    diag_1 = np.eye(2 * batch_size, k=-batch_size)
    diag_2 = np.eye(2 * batch_size, k=batch_size)

    mask = diag_0 + diag_1 + diag_2
    mask = 1 - mask
    mask = torch.from_numpy(mask)
    mask = mask.to(device, dtype=torch.bool)
    return mask


class NTXentLossPoly(nn.Module):
    def __init__(self, temperature=0.2, is_cosine=True):
        r"""
        modified from the implementation of NTXentLoss_poly from
        https://github.com/mims-harvard/TFC-pretraining
        """
        super(NTXentLossPoly, self).__init__()
        self.temperature = temperature
        self.is_cosine = is_cosine

    def _get_similarity(self, data):
        is_cosine = self.is_cosine
        if is_cosine:
            return _cosine_similarity(data)
        return _dot_similarity(data)

    def forward(self, data_i, data_j):
        batch_size = data_i.size()[0]
        device = data_i.device

        data = torch.cat((data_i, data_j, ), dim=0)
        similarity = self._get_similarity(data)

        positive_upper = torch.diag(similarity, batch_size)
        positive_lower = torch.diag(similarity, -batch_size)
        positive = torch.cat((positive_upper, positive_lower, ), dim=0)
        positive = positive.unsqueeze(1)

        negative_mask = _get_mask(batch_size, device)
        negative = similarity[negative_mask].view(
            2 * batch_size, 2 * batch_size - 2)

        logits = torch.cat((positive, negative), dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(2 * batch_size)
        labels = labels.to(device, dtype=torch.long)
        cross_entropy = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

        labels_onthot = torch.zeros((2 * batch_size, 2 * batch_size - 1))
        labels_onthot[:, 0] = 1
        labels_onthot = labels_onthot.to(device)
        poly_loss = torch.mean(labels_onthot * nn.Softmax(dim=-1)(logits))

        loss = (cross_entropy / (2 * batch_size) +
                batch_size * (1 / batch_size - poly_loss))
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2, is_cosine=True):
        r"""
        modified from the implementation of NTXentLoss from
        https://github.com/mims-harvard/TFC-pretraining
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.is_cosine = is_cosine

    def _get_similarity(self, data):
        is_cosine = self.is_cosine
        if is_cosine:
            return _cosine_similarity(data)
        return _dot_similarity(data)

    def forward(self, data_i, data_j):
        batch_size = data_i.size()[0]
        device = data_i.device

        data = torch.cat((data_i, data_j, ), dim=0)
        similarity = self._get_similarity(data)

        positive_upper = torch.diag(similarity, batch_size)
        positive_lower = torch.diag(similarity, -batch_size)
        positive = torch.cat((positive_upper, positive_lower, ), dim=0)
        positive = positive.unsqueeze(1)

        negative_mask = _get_mask(batch_size, device)
        negative = similarity[negative_mask].view(
            2 * batch_size, 2 * batch_size - 2)

        logits = torch.cat((positive, negative), dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(2 * batch_size)
        labels = labels.to(device, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        return loss / (2 * batch_size)
