# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import argparse
import pathlib
from random import shuffle
from engine.util import parse_config
from engine.data_io import get_ucr_data_names as get_data_names
from engine.data_io import load_ucr_dataset as load_dataset
from engine.train_test import dist_eval


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        default='ucr_00')
    parser.add_argument('--method_name')
    parser.add_argument('--dataset_order',
                        type=int, choices=[-1, 0, 1, ], default=1)

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    dataset_order = args.dataset_order

    main(data_name, method_name, dataset_order)


def main(data_config_name, method_name, dataset_order):
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

    for dataset_name in dataset_names:
        result_dir = os.path.join(
            '.', 'result_agg', f'{data_config_name}_{dataset_name}')
        path = pathlib.Path(result_dir)
        path.mkdir(parents=True, exist_ok=True)

        result_path = os.path.join(
            result_dir, f'{method_name}.npz')

        dataset = load_dataset(dataset_name, data_config)
        dist_eval(dataset, method_config, result_path)


if __name__ == '__main__':
    main_wrapper()

