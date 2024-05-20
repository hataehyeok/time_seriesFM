# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
from .util import write_config


def get_dist_classifier():
    config_dict = {}
    config_dict['model'] = {'metric': 'ed'}
    config_path = os.path.join(
        '..', 'file', 'dist_0000.config')
    write_config(config_dict, config_path)

    config_dict = {}
    config_dict['model'] = {'metric': 'dtw'}
    config_path = os.path.join(
        '..', 'file', 'dist_0001.config')
    write_config(config_dict, config_path)

