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
from engine.data_io import load_ucr_pretrain as load_dataset
from engine.model import get_model
from engine.train_test import nn_pretrain


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        default='ucr_00')
    parser.add_argument('--method_name')

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name

    main(data_name, method_name)


def main(data_config_name, method_name):
    data_config = os.path.join(
        '.', 'config', 'file', f'{data_config_name}.config')
    data_config = parse_config(data_config)

    method_config = os.path.join(
        '.', 'config', 'file', f'{method_name}.config')
    method_config = parse_config(method_config)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_dir = os.path.join(
        '.', 'model', f'{data_config_name}_pretrain')
    path = pathlib.Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)

    fmt_str = '{0:04d}'
    model_path = os.path.join(
        model_dir, f'{method_name}_{fmt_str}.npz')

    dataset = load_dataset(data_config)
    method_config['in_dim'] = dataset.shape[1]
    method_config['data_len'] = dataset.shape[2]
    model = get_model(method_config)
    nn_pretrain(dataset, model, model_path,
                method_config['train'], device)


if __name__ == '__main__':
    main_wrapper()

