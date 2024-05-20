# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .util import write_config


def get_dataset(ucr_dir):
    config_dict = {}
    config_dict['data'] = {}
    config_dict['data']['data_dir'] = ucr_dir
    config_dict['data']['max_len'] = 512
    config_dict['data']['seed'] = 666
    config_dict['data']['pretrain_frac'] = 0.5
    config_dict['data']['train_frac'] = 0.3
    config_dict['data']['valid_frac'] = 0.1
    config_dict['data']['test_frac'] = 0.1
    config_dict['data']['is_same_length'] = 'True'

    config_path = os.path.join(
        '..', 'file', 'ucr_00.config')
    write_config(config_dict, config_path)

