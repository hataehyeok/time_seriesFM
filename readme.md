# Finding Optimal Head on Foundation Model for Time-Series Classification

Extended research of 'Toward a Foundation Model for Time Series Data, CIKM'23'

## Environment
The code has only been tested with the environment list below:
* python=3.8.13
* numpy=1.19.5
* torch=1.10.2+cu102
* tslearn=0.5.2
* scipy=1.6.2

```sh
conda install numpy=1.19.5
conda install scipy=1.6.2
pip install torch==1.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
conda install -c conda-forge tslearn=0.5.2

```

## Verify the installed version

```sh
# Check Python version
python --version

# Check numpy version
python -c "import numpy; print('numpy:', numpy.__version__)"

# Check torch version
python -c "import torch; print('torch:', torch.__version__)"

# Check tslearn version
python -c "import tslearn; print('tslearn:', tslearn.__version__)"

# Check scipy version
python -c "import scipy; print('scipy:', scipy.__version__)"

```


## Procedure
Please follow the steps below to reproduce the experiments.
1. download the UCR archive from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and unzip the dataset
2. change line 16 in ```./config/script/script_config_0.py``` to the path of the directory that contains the UCR archive
3. change the current folder to ```./config/script/```
4. run ```python script_config_0.py``` to generate all the config files
3. change the current folder back to ```.```
4. run the following commands to test Euclidean distance and dynamic time warping baselines:
    * ```python script_ucr_dist_0.py --method_name dist_0000``` for Euclidean distance
    * ```python script_ucr_dist_0.py --method_name dist_0001``` for dynamic time warping
5. run the following commands to train/test non-pre-trained models:
    * ```python script_ucr_nn_0.py --method_name lst_c_0000``` for long short-term memory network
    * ```python script_ucr_nn_0.py --method_name gru_c_0000``` for gated recurrent unit network
    * ```python script_ucr_nn_0.py --method_name r1d_c_0000``` for residual network
    * ```python script_ucr_nn_0.py --method_name trf_c_0000``` for transformer
6. run the following commands to pre-train models:
    * ```python script_ucr_pretrain_0.py --method_name lst_sc_0000``` for long short-term memory network + SimCLR
    * ```python script_ucr_pretrain_0.py --method_name lst_tv_0000``` for long short-term memory network + TS2Vec
    * ```python script_ucr_pretrain_0.py --method_name lst_mu_0000``` for long short-term memory network + MixingUp
    * ```python script_ucr_pretrain_0.py --method_name lst_tf_0000``` for long short-term memory network + TF-C
    * ```python script_ucr_pretrain_0.py --method_name lst_tc_0000``` for long short-term memory network + TimeCLR (the proposed method)
    * ```python script_ucr_pretrain_0.py --method_name gru_sc_0000``` for gated recurrent unit network + SimCLR
    * ```python script_ucr_pretrain_0.py --method_name gru_tv_0000``` for gated recurrent unit network + TS2Vec
    * ```python script_ucr_pretrain_0.py --method_name gru_mu_0000``` for gated recurrent unit network + MixingUp
    * ```python script_ucr_pretrain_0.py --method_name gru_tf_0000``` for gated recurrent unit network + TF-C
    * ```python script_ucr_pretrain_0.py --method_name gru_tc_0000``` for gated recurrent unit network + TimeCLR (the proposed method)
    * ```python script_ucr_pretrain_0.py --method_name r1d_sc_0000``` for residual network + SimCLR
    * ```python script_ucr_pretrain_0.py --method_name r1d_tv_0000``` for residual network + TS2Vec
    * ```python script_ucr_pretrain_0.py --method_name r1d_mu_0000``` for residual network + MixingUp
    * ```python script_ucr_pretrain_0.py --method_name r1d_tf_0000``` for residual network + TF-C
    * ```python script_ucr_pretrain_0.py --method_name r1d_tc_0000``` for residual network + TimeCLR (the proposed method)
    * ```python script_ucr_pretrain_0.py --method_name trf_sc_0000``` for transformer + SimCLR
    * ```python script_ucr_pretrain_0.py --method_name trf_tv_0000``` for transformer + TS2Vec
    * ```python script_ucr_pretrain_0.py --method_name trf_mu_0000``` for transformer + MixingUp
    * ```python script_ucr_pretrain_0.py --method_name trf_tf_0000``` for transformer + TF-C
    * ```python script_ucr_pretrain_0.py --method_name trf_tc_0000``` for transformer + TimeCLR (the proposed method)
6. run the following commands to fine-tune/test models:
    * ```python script_ucr_nn_0.py --method_name lst_sc_c_0000``` for long short-term memory network + SimCLR
    * ```python script_ucr_nn_0.py --method_name lst_tv_c_0000``` for long short-term memory network + TS2Vec
    * ```python script_ucr_nn_0.py --method_name lst_mu_c_0000``` for long short-term memory network + MixingUp
    * ```python script_ucr_nn_0.py --method_name lst_tf_c_0000``` for long short-term memory network + TF-C
    * ```python script_ucr_nn_0.py --method_name lst_tc_c_0000``` for long short-term memory network + TimeCLR (the proposed method)
    * ```python script_ucr_nn_0.py --method_name gru_sc_c_0000``` for gated recurrent unit network + SimCLR
    * ```python script_ucr_nn_0.py --method_name gru_tv_c_0000``` for gated recurrent unit network + TS2Vec
    * ```python script_ucr_nn_0.py --method_name gru_mu_c_0000``` for gated recurrent unit network + MixingUp
    * ```python script_ucr_nn_0.py --method_name gru_tf_c_0000``` for gated recurrent unit network + TF-C
    * ```python script_ucr_nn_0.py --method_name gru_tc_c_0000``` for gated recurrent unit network + TimeCLR (the proposed method)
    * ```python script_ucr_nn_0.py --method_name r1d_sc_c_0000``` for residual network + SimCLR
    * ```python script_ucr_nn_0.py --method_name r1d_tv_c_0000``` for residual network + TS2Vec
    * ```python script_ucr_nn_0.py --method_name r1d_mu_c_0000``` for residual network + MixingUp
    * ```python script_ucr_nn_0.py --method_name r1d_tf_c_0000``` for residual network + TF-C
    * ```python script_ucr_nn_0.py --method_name r1d_tc_c_0000``` for residual network + TimeCLR (the proposed method)
    * ```python script_ucr_nn_0.py --method_name trf_sc_c_0000``` for transformer + SimCLR
    * ```python script_ucr_nn_0.py --method_name trf_tv_c_0000``` for transformer + TS2Vec
    * ```python script_ucr_nn_0.py --method_name trf_mu_c_0000``` for transformer + MixingUp
    * ```python script_ucr_nn_0.py --method_name trf_tf_c_0000``` for transformer + TF-C
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000``` for transformer + TimeCLR (the proposed method)
5. run ```python script_result_0.py``` to get the experiment results.

