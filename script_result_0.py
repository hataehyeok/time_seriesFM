# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import numpy as np
import scipy.stats
import csv
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
    for method_name in method_names:
        is_method_done = True
        for data_name in data_names:
            result_agg_dir = os.path.join('.', 'result_agg', f'{data_config_name}_{data_name}')
            result_agg_path = os.path.join(result_agg_dir, method_name)
            if not os.path.isfile(result_agg_path):
                is_method_done = False
                n_not_done += 1
                break

        if not is_method_done:
            print(f'{method_name} not done!')
            is_done = False
    return is_done, n_not_done


def print_table_to_csv(method_names, data_config_name, get_data_names, csv_file_path, print_dataset=True):
    is_done, n_not_done = check_not_done(method_names, data_config_name, get_data_names)
    if not is_done:
        print(f'total of {n_not_done:d} methods are not done!')
        return

    data_names = get_data_names()
    mean_rank = []

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([''] + method_names)

        for data_name in data_names:
            result_dir = os.path.join(
                '.', 'result_agg', f'{data_config_name}_{data_name}')
            acc_tests = []
            for method_name in method_names:
                result_path = os.path.join(result_dir, method_name)
                acc_test, _ = get_acc(result_path)
                acc_tests.append(acc_test)

            rank_tests = scipy.stats.rankdata(-np.array(acc_tests))
            mean_rank.append(rank_tests)
            if print_dataset:
                csvwriter.writerow([data_name] + acc_tests)

        mean_rank = np.array(mean_rank)
        mean_rank = np.mean(mean_rank, axis=0)
        csvwriter.writerow(['Average Rank'] + list(mean_rank))


def main():
    method_names = [
        'trf_tc_c_0000_freeze_class_token_gt.npz',
        'trf_tc_c_0000_freeze_flatten_gt.npz',
        'trf_tc_c_0000_freeze_pooling_gt.npz',
        'trf_tc_c_0000_freeze_pooling_st.npz',
        'trf_tc_c_0000_freeze_pooling_dt.npz',
        'trf_tc_c_0000_nofreeze_class_token_gt.npz',
        'trf_tc_c_0000_nofreeze_flatten_gt.npz',
        'trf_tc_c_0000_nofreeze_pooling_gt.npz',
        'trf_tc_c_0000_nofreeze_pooling_st.npz',
        'trf_tc_c_0000_nofreeze_pooling_dt.npz',
    ]
    
    print_dataset = True
    get_data_names = get_ucr_data_names
    print_table_to_csv(method_names, 'ucr_00', get_data_names, 'test_dt_seg_9.csv', print_dataset=print_dataset)


if __name__ == '__main__':
    main()
