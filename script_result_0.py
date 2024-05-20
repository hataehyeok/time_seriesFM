# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import numpy as np
import scipy.stats
from engine.data_io import get_ucr_data_names


def get_acc(result_agg_path):
    pkl = np.load(result_agg_path)
    acc_valid = pkl['acc_valid']
    acc_test = pkl['acc_test']
    return acc_test, acc_valid


def check_not_done(method_names, data_config_name, get_data_names):
    is_done = True
    n_not_done = 0
    data_names = get_data_names()
    for method_name, _ in method_names:
        is_method_done = True
        for data_name in data_names:
            result_agg_dir = os.path.join(
                '.', 'result_agg', f'{data_config_name}_{data_name}')
            result_agg_path = os.path.join(
                result_agg_dir, f'{method_name}.npz')
            if not os.path.isfile(result_agg_path):
                is_method_done = False
                n_not_done += 1
                break

        if not is_method_done:
            print(f'{method_name} not done!')
            is_done = False
    return is_done, n_not_done


def print_table(method_names, data_config_name, get_data_names,
                base_map, print_dataset=True):
    is_done, n_not_done = check_not_done(
        method_names, data_config_name, get_data_names)
    if not is_done:
        print(f'total of {n_not_done:d} methods are not done!')
        return

    print(',', end='')
    for _, method_name in method_names:
        print(f'{method_name},', end='')
    print()

    data_names = get_data_names()
    mean_rank = []
    for data_name in data_names:
        result_dir = os.path.join(
            '.', 'result_agg', f'{data_config_name}_{data_name}')
        if print_dataset:
            print(f'{data_name},', end='')
        acc_tests = []
        for method_name, _ in method_names:
            result_path = os.path.join(
                result_dir, f'{method_name}.npz')
            if 'dist' in method_name:
                acc_test, _ = get_acc(result_path)
            else:
                acc_test_0, acc_valid_0 = get_acc(
                    result_path)
                arch = method_name[:3]
                base_name = base_map[arch]
                result_path = os.path.join(
                    result_dir, f'{base_name}.npz')
                acc_test_1, acc_valid_1 = get_acc(
                    result_path)
                if acc_valid_0 > acc_valid_1:
                    acc_test = acc_test_0
                else:
                    acc_test = acc_test_1

            acc_tests.append(acc_test)

        rank_tests = scipy.stats.rankdata(-np.array(acc_tests))
        mean_rank.append(rank_tests)
        if print_dataset:
            for acc_test in acc_tests:
                print(f'{acc_test:f},', end='')
            print()

    mean_rank = np.array(mean_rank)
    mean_rank = np.mean(mean_rank, axis=0)
    print(f'Average Rank,', end='')
    for val in mean_rank:
        print(f'{val:f},', end='')
    print()


def main():
    arcts = ['lst', 'gru', 'r1d', 'trf', ]
    arct_names = ['LSTM', 'GRU', 'ResNet', 'XFMR', ]
    base_map = {'gru': 'gru_c_0000', 'lst': 'lst_c_0000',
                'r1d': 'r1d_c_0000', 'trf': 'trf_c_0000', }

    pt_tamps = ['_sc_c_0000', '_tv_c_0000',
                '_mu_c_0000', '_tf_c_0000', '_tc_c_0000', ]
    pt_names = ['SimCLR', 'TS2Vec', 'MixingUp',  'TF-C',  'TimeCLR', ]

    method_names = [
        ['dist_0000', 'ED', ],
        ['dist_0001', 'DTW', ],
    ]

    for arct, arct_name in zip(arcts, arct_names):
        method_names.append([base_map[arct], arct_name, ])
        for pt_tamp, pt_name in zip(pt_tamps, pt_names):
            method_names.append([arct + pt_tamp, pt_name, ])

    print_dataset = True
    get_data_names = get_ucr_data_names
    print_table(method_names, 'ucr_00', get_data_names,
                base_map, print_dataset=print_dataset)


if __name__ == '__main__':
    main()

