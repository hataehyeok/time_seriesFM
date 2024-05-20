# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .base_setting import get_gru_setting
from .base_setting import get_lst_setting
from .base_setting import get_r1d_setting
from .base_setting import get_trf_setting
from .base_setting import get_classifier_setting
from .base_setting import get_train_setting
from .util import write_config


def get_base_classifier():
    # Layer Normalization
    # Set classifier for each model
    norm = 'LN'

    prefix = 'gru_c'
    config_id = 0

    config_dict = {}
    config_dict['model'] = {'model_name': 'classifier_rnnet'}
    config_dict['classifier'] = get_classifier_setting()
    config_dict['encoder'] = get_gru_setting()
    config_dict['train'] = get_train_setting()

    config_dict['encoder']['is_projector'] = 'True'
    config_dict['encoder']['project_norm'] = norm

    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)

    prefix = 'lst_c'
    config_id = 0

    config_dict = {}
    config_dict['model'] = {'model_name': 'classifier_rnnet'}
    config_dict['classifier'] = get_classifier_setting()
    config_dict['encoder'] = get_lst_setting()
    config_dict['train'] = get_train_setting()

    config_dict['encoder']['is_projector'] = 'True'
    config_dict['encoder']['project_norm'] = norm

    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)

    prefix = 'r1d_c'
    config_id = 0

    config_dict = {}
    config_dict['model'] = {'model_name': 'classifier_resnet1d'}
    config_dict['classifier'] = get_classifier_setting()
    config_dict['encoder'] = get_r1d_setting()
    config_dict['train'] = get_train_setting()

    config_dict['encoder']['is_projector'] = 'True'
    config_dict['encoder']['project_norm'] = norm
    config_dict['encoder']['norm'] = norm

    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)

    prefix = 'trf_c'
    config_id = 0

    config_dict = {}
    config_dict['model'] = {'model_name': 'classifier_transform'}
    config_dict['classifier'] = get_classifier_setting()
    config_dict['encoder'] = get_trf_setting()
    config_dict['train'] = get_train_setting()

    config_dict['encoder']['is_projector'] = 'True'
    config_dict['encoder']['project_norm'] = norm

    config_path = os.path.join(
        '..', 'file', f'{prefix}_{config_id:04d}.config')

    config_id += 1
    write_config(config_dict, config_path)

