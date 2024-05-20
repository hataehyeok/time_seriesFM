# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import configparser
import numpy as np
from collections import OrderedDict


def parse_config(config_path, verbose=True):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if verbose:
        print(config_path)
    config_dict = OrderedDict()
    for key_0 in parser:
        config_dict[key_0] = OrderedDict()
        for key_1 in parser[key_0]:
            val = parser[key_0][key_1]
            if val == 'None':
                val = None
            config_dict[key_0][key_1] = val
            if verbose:
                print(f'  {key_0}.{key_1}={val}')
    return config_dict


def get_agg_result(result_path, result_agg_path, train_config):
    if os.path.isfile(result_agg_path):
        pkl = np.load(result_agg_path)
        epoch = pkl['epoch']
        return epoch

    n_epoch = int(train_config['n_epoch'])
    acc_valid_bsf = 0
    for i in range(n_epoch):
        result_path_ = result_path.format(i)
        if not os.path.isfile(result_path_):
            return

        pkl = np.load(result_path_)
        acc_valid = pkl['acc_valid']
        acc_test = pkl['acc_test']

        if acc_valid_bsf == 0:
            acc_valid_bsf = acc_valid
            acc_test_bsf = acc_test
            epoch_bsf = i
        elif acc_valid > acc_valid_bsf:
            acc_valid_bsf = acc_valid
            acc_test_bsf = acc_test
            epoch_bsf = i
    np.savez(result_agg_path,
             acc_valid=acc_valid_bsf,
             acc_test=acc_test_bsf,
             epoch=epoch_bsf)

