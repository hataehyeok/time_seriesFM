# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import time
import numpy as np
import torch
import torch.nn as nn
from .util import _get_checkpoint
from .util import _get_start_epoch
from .util import _normalize_dataset


def nn_train(dataset, model, model_path,
             train_config, device):
    data = dataset['data_train']
    label = dataset['label_train']

    data = _normalize_dataset(data)

    model.to(device)
    model.train()

    # model.freeze_backbone()

    lr = float(train_config['lr'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr)

    n_data = data.shape[0]
    batch_size = int(train_config['batch_size'])
    n_iter = np.ceil(n_data / batch_size)
    n_iter = int(n_iter)
    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])

    ckpts = _get_checkpoint(n_ckpt, n_epoch)
    start_epoch = _get_start_epoch(model_path, ckpts)

    # For saving loss and time
    loss_train = np.zeros(n_epoch)
    toc_train = np.zeros(n_epoch)

    for i in range(start_epoch, n_epoch):
        if start_epoch != 0 and i == start_epoch:
            print(f'resume training from epoch {i + 1:d}')
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            print(f'loading {model_path_i}')
            pkl = torch.load(model_path_i, map_location='cpu')
            loss_train = pkl['loss_train']
            toc_train = pkl['toc_train']
            loss_epoch = loss_train[i]
            toc_epoch = toc_train[i]

            model.load_state_dict(
                pkl['model_state_dict'])
            model.to(device)
            model.train()

            optimizer.load_state_dict(
                pkl['optimizer_state_dict'])
            print((f'epoch {i + 1}/{n_epoch}, '
                   f'loss={loss_epoch:0.4f}, '
                   f'time={toc_epoch:0.2f}.'))
            continue

        tic = time.time()
        loss_epoch = 0
        idx_order = np.random.permutation(n_data)
        for j in range(n_iter):
            optimizer.zero_grad()

            idx_start = j * batch_size
            idx_end = (j + 1) * batch_size
            if idx_end > n_data:
                idx_end = n_data
            idx_batch = idx_order[idx_start:idx_end]

            batch_size_ = idx_end - idx_start
            if batch_size_ < batch_size:
                n_fill = batch_size - batch_size_
                idx_fill = idx_order[:n_fill]
                idx_batch = np.concatenate(
                    (idx_batch, idx_fill, ), axis=0)

            data_batch = data[idx_batch, :, :]
            label_batch = label[idx_batch]

            label_batch = torch.from_numpy(label_batch)
            label_batch = label_batch.to(device, dtype=torch.long)

            logit = model.forward(
                data_batch, normalize=False, to_numpy=False)

            loss = nn.CrossEntropyLoss()(logit, label_batch)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= n_iter
        toc_epoch = time.time() - tic

        loss_train[i] = loss_epoch
        toc_train[i] = toc_epoch
        if i in ckpts:
            pkl = {}
            pkl['loss_train'] = loss_train
            pkl['toc_train'] = toc_train
            pkl['model_state_dict'] = model.state_dict()
            pkl['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(pkl, model_path_i)

        print((f'epoch {i + 1}/{n_epoch}, '
               f'loss={loss_epoch:0.4f}, '
               f'time={toc_epoch:0.2f}.'))

