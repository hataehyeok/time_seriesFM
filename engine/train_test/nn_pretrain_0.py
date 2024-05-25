# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import time
import copy
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
from .util import _get_checkpoint
from .util import _normalize_dataset
from .loss import NTXentLossPoly
from .loss import NTXentLoss
from .loss import HierContrastLoss
from .loss import MixupLoss


def _get_start_epoch(model_path, ckpts):
    start_epoch = 0
    for i in ckpts:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            start_epoch = i

    model_path_i = model_path.format(start_epoch)
    if not os.path.isfile(model_path_i):
        return start_epoch

    try:
        pkl = torch.load(model_path_i, map_location='cpu')
    except:
        print(f'{model_path_i} can not be opened. It is removed!')
        os.remove(model_path_i)
        start_epoch = _get_start_epoch(model_path, ckpts)
    return start_epoch


def _timefreq_encoder_forward(model, idx_batch, data):
    data_batch = data[idx_batch, :, :]
    h_t, z_t, h_f, z_f = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=False)
    h_t_aug, z_t_aug, h_f_aug, z_f_aug = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = NTXentLossPoly()
    loss_t = loss_fun(h_t, h_t_aug)
    loss_f = loss_fun(h_f, h_f_aug)
    loss_tf = loss_fun(z_t, z_f)
    loss = 0.2 * (loss_t + loss_f) + loss_tf
    return loss


def _simclr_encoder_forward(model, idx_batch, data):
    data_batch = data[idx_batch, :, :]
    ts_emb_aug_0 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    ts_emb_aug_1 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = NTXentLoss()
    loss = loss_fun(ts_emb_aug_0, ts_emb_aug_1)
    return loss


def _timeclr_encoder_forward(model, idx_batch, data):
    data_batch = data[idx_batch, :, :]
    # First augmented embedding
    ts_emb_aug_0 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    # Second augmented embedding
    ts_emb_aug_1 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = NTXentLossPoly()
    # Contrastive loss
    loss = loss_fun(ts_emb_aug_0, ts_emb_aug_1)
    return loss


def _ts2vec_encoder_forward(model, idx_batch, data):
    data_batch = data[idx_batch, :, :]
    ts_emb_l, ts_emb_r = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = HierContrastLoss()
    loss = loss_fun(ts_emb_l, ts_emb_r)
    return loss


def _mixup_encoder_forward(model, idx_batch, data):
    data_batch = data[idx_batch, :, :]
    ts_emb_0, ts_emb_1, ts_emb_aug, lam = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = MixupLoss()
    loss = loss_fun(ts_emb_0, ts_emb_1, ts_emb_aug, lam)
    return loss


def nn_pretrain(data, model, model_path, train_config, device):
    model.to(device)
    model.train()

    pretrain_name = model.pretrain_name
    lr = float(train_config['lr'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr * 100)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', verbose=True)

    n_data = data.shape[0]
    batch_size = int(train_config['batch_size'])
    n_iter = np.ceil(n_data / batch_size)
    n_iter = int(n_iter)
    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])

    ckpts = _get_checkpoint(n_ckpt, n_epoch)
    start_epoch = _get_start_epoch(model_path, ckpts)

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', verbose=True)
            print((f'epoch {i + 1}/{n_epoch}, '
                   f'loss={loss_epoch:0.4f}, '
                   f'time={toc_epoch:0.2f}.'))
            continue

        model_state_dict_old = copy.deepcopy(
            model.state_dict())
        optimizer_state_dict_old = copy.deepcopy(
            optimizer.state_dict())
        while True:
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

                pretrain_name = model.pretrain_name
                if pretrain_name == 'timefreq':
                    loss = _timefreq_encoder_forward(
                        model, idx_batch, data)
                elif pretrain_name == 'ts2vec':
                    loss = _ts2vec_encoder_forward(
                        model, idx_batch, data)
                elif pretrain_name == 'mixup':
                    loss = _mixup_encoder_forward(
                        model, idx_batch, data)
                elif pretrain_name == 'simclr':
                    loss = _simclr_encoder_forward(
                        model, idx_batch, data)
                elif pretrain_name == 'timeclr':
                    loss = _timeclr_encoder_forward(
                        model, idx_batch, data)
                else:
                    raise Exception(
                        f'unknown pretrain name: {pretrain_name}')

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

            if np.isfinite(loss_epoch):
                break
            else:
                print('restart model training...')
                model.load_state_dict(
                    model_state_dict_old)
                model.to(device)
                model.train()

                optimizer.load_state_dict(
                    optimizer_state_dict_old)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', verbose=True)

        scheduler.step(loss_epoch)

