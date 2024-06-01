#!/bin/bash

# Function to run a command with specified GPUs
run_command() {
    local gpu_ids=$1
    local command=$2
    local log_file=$3
    CUDA_VISIBLE_DEVICES=${gpu_ids} nohup ${command} > "${log_file}" 2>&1 &
}

# Commands to run
commands=(
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode class_token"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode flatten"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode gt"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode st"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode dt"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode class_token"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode flatten"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode gt"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode st"
    "python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode dt"
)

# Assign GPU IDs
gpu_ids=("0" "1" "2")

# Run commands in parallel
for i in ${!commands[@]}; do
    gpu_index=$((i % 3))
    log_file="log_${i}.txt"
    run_command ${gpu_ids[$gpu_index]} "${commands[$i]}" "${log_file}"
done

# Wait for all background jobs to finish
wait
