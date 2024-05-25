#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2

# Execute Python scripts with different parameters
python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode class_token

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode flatten

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode gt

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode st

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode class_token

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode flatten

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode gt

python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode st
