# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .base_setting import get_gru_setting
from .base_setting import get_lst_setting
from .base_setting import get_r1d_setting
from .base_setting import get_trf_setting
from .base_setting import get_timefreq_setting
from .base_setting import get_ts2vec_setting
from .base_setting import get_mixup_setting
from .base_setting import get_simclr_setting
from .base_setting import get_timeclr_setting
from .base_setting import get_classifier_setting
from .base_setting import get_train_setting
from .util import write_config

# Configure encoder settings according to the pre-training method and create a config dict
# That config file is used to generate the pretrained model

def _get_pretrain_setting(short_name, setting_fun, setting_str,
                          pretrain_setting):
    prefix = f'{short_name}_{pretrain_setting[0]}'
    model_name = f'{pretrain_setting[1]}_{setting_str}'

    config_id = 0
    batch_size = 256
    norm = 'LN'
    config_dict = {}
    config_dict['model'] = {'model_name': model_name}

    config_dict['encoder'] = setting_fun()
    if 'norm' in config_dict['encoder']:
        config_dict['encoder']['norm'] = norm
    config_dict['encoder']['in_dim'] = 1

    if pretrain_setting[1] == 'timefreq':
        if 'out_dim' in config_dict['encoder']:
            config_dict['encoder']['out_dim'] = int(
                config_dict['encoder']['out_dim'] / 2)
        if 'n_dim' in config_dict['encoder']:
            config_dict['encoder']['n_dim'] = int(
                config_dict['encoder']['n_dim'] / 2)
        config_dict['timefreq'] = get_timefreq_setting()
        config_dict['timefreq']['project_norm'] = norm

    elif pretrain_setting[1] == 'ts2vec':
        config_dict['ts2vec'] = get_ts2vec_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'mixup':
        config_dict['mixup'] = get_mixup_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'simclr':
        config_dict['simclr'] = get_simclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'timeclr':
        config_dict['timeclr'] = get_timeclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    config_dict['train'] = get_train_setting()
    config_dict['train']['n_ckpt'] = 100
    config_dict['train']['batch_size'] = batch_size

    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)


# Config file for classifier
# Classification: Train the model for a specific task by fetching the weights of a pre-trained model

def _get_classifier_setting(short_name, setting_fun, setting_str,
                            pretrain_setting):
    prefix = f'{short_name}_{pretrain_setting[0]}_c'
    model_name = f'classifier_{pretrain_setting[1]}_{setting_str}'

    config_id = 0
    batch_size = 256
    norm = 'LN'
    pretrain_data = 'ucr_00_pretrain'
    config_dict = {}
    config_dict['model'] = {'model_name': model_name}

    config_dict['classifier'] = get_classifier_setting()

    config_dict['encoder'] = setting_fun()
    if 'norm' in config_dict['encoder']:
        config_dict['encoder']['norm'] = norm
    config_dict['encoder']['in_dim'] = 1

    if pretrain_setting[1] == 'timefreq':
        if 'out_dim' in config_dict['encoder']:
            config_dict['encoder']['out_dim'] = int(
                config_dict['encoder']['out_dim'] / 2)
        if 'n_dim' in config_dict['encoder']:
            config_dict['encoder']['n_dim'] = int(
                config_dict['encoder']['n_dim'] / 2)
        config_dict['timefreq'] = get_timefreq_setting()
        config_dict['timefreq']['project_norm'] = norm
        config_dict['timefreq']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tf_0000_0399.npz'))

    elif pretrain_setting[1] == 'ts2vec':
        config_dict['ts2vec'] = get_ts2vec_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['ts2vec']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tv_0000_0399.npz'))

    elif pretrain_setting[1] == 'mixup':
        config_dict['mixup'] = get_mixup_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['mixup']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_mu_0000_0399.npz'))

    elif pretrain_setting[1] == 'simclr':
        config_dict['simclr'] = get_simclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['simclr']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_sc_0000_0399.npz'))

    elif pretrain_setting[1] == 'timeclr':
        config_dict['timeclr'] = get_timeclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['timeclr']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tc_0000_0399.npz'))

    config_dict['train'] = get_train_setting()
    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)


def get_pretrain_model():
    pretrain_settings = [
        ['tf', 'timefreq', ],
        ['tv', 'ts2vec', ],
        ['mu', 'mixup', ],
        ['sc', 'simclr', ],
        ['tc', 'timeclr', ],
    ]

    for pretrain_setting in pretrain_settings:
        _get_pretrain_setting(
            'gru', get_gru_setting,
            'rnnet', pretrain_setting)
        _get_pretrain_setting(
            'lst', get_lst_setting,
            'rnnet', pretrain_setting)
        _get_pretrain_setting(
            'r1d', get_r1d_setting,
            'resnet1d', pretrain_setting)
        _get_pretrain_setting(
            'trf', get_trf_setting,
            'transform', pretrain_setting)

    for pretrain_setting in pretrain_settings:
        _get_classifier_setting(
            'gru', get_gru_setting,
            'rnnet', pretrain_setting)
        _get_classifier_setting(
            'lst', get_lst_setting,
            'rnnet', pretrain_setting)
        _get_classifier_setting(
            'r1d', get_r1d_setting,
            'resnet1d', pretrain_setting)
        _get_classifier_setting(
            'trf', get_trf_setting,
            'transform', pretrain_setting)

