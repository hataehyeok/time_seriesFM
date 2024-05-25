# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import copy
import argparse
import pathlib
import torch
from random import shuffle
from engine.util import parse_config
from engine.util import get_agg_result
from engine.data_io import get_ucr_data_names as get_data_names
from engine.data_io import load_ucr_dataset as load_dataset
from engine.model import get_model
from engine.train_test import nn_train
from engine.train_test import nn_eval


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        default='ucr_00')
    parser.add_argument('--method_name')
    parser.add_argument('--dataset_order',
                        type=int, choices=[-1, 0, 1, ], default=1)
    parser.add_argument('--is_freeze', type=lambda s:s.lower() in ['true','1'],
                        default=False, help='is_freeze (default: False)')
    parser.add_argument('--aggregation_mode', type=str, default='class_token')
    parser.add_argument('--pooling_mode', type=str, default='gt')

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    dataset_order = args.dataset_order

    main(data_name, method_name, dataset_order, args.is_freeze, args.aggregation_mode, args.pooling_mode)


def main(data_config_name, method_name, dataset_order, is_freeze, aggregation_mode, pooling_mode):
    data_config = os.path.join(
        '.', 'config', 'file', f'{data_config_name}.config')
    data_config = parse_config(data_config)

    method_config = os.path.join(
        '.', 'config', 'file', f'{method_name}.config')
    method_config = parse_config(method_config)

    dataset_names = get_data_names()
    if dataset_order == -1:
        dataset_names = dataset_names[::-1]
    elif dataset_order == 0:
        shuffle(dataset_names)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    fmt_str = '{0:04d}'
    for dataset_name in dataset_names:
        result_dir = os.path.join(
            '.', 'result', f'{data_config_name}_{dataset_name}')
        result_agg_dir = os.path.join(
            '.', 'result_agg', f'{data_config_name}_{dataset_name}')
        model_dir = os.path.join(
            '.', 'model', f'{data_config_name}_{dataset_name}')

        path = pathlib.Path(result_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(result_agg_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        freeze_str = 'freeze' if is_freeze else 'nofreeze'
        agg_mode_str = f'{aggregation_mode}'
        pooling_mode_str = f'{pooling_mode}'

        result_path = os.path.join(
            result_dir, f'{method_name}_{freeze_str}_{agg_mode_str}_{pooling_mode_str}_{fmt_str}.npz')
        result_agg_path = os.path.join(
            result_agg_dir, f'{method_name}_{freeze_str}_{agg_mode_str}_{pooling_mode_str}.npz')
        model_path = os.path.join(
            model_dir, f'{method_name}_{freeze_str}_{agg_mode_str}_{pooling_mode_str}_{fmt_str}.npz')
        
        if os.path.isfile(result_agg_path):
            continue

        dataset = load_dataset(dataset_name, data_config)
        method_config_ = copy.deepcopy(method_config)
        method_config_['in_dim'] = dataset['n_dim']
        method_config_['n_class'] = dataset['n_class']
        method_config_['data_len'] = dataset['data_len']
        model = get_model(method_config_, aggregation_mode, pooling_mode)
        nn_train(dataset, model, model_path,
                 method_config_['train'], device, is_freeze=is_freeze)
        nn_eval(dataset, model, model_path,
                result_path, method_config_['train'], device)
        get_agg_result(result_path, result_agg_path,
                       method_config_['train'])


if __name__ == '__main__':
    main_wrapper()

