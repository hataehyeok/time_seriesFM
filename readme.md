# Finding Effective Aggregation on Foundation Model for Time-Series Classification

Extended research of 'Toward a Foundation Model for Time Series Data, CIKM'23'

## Environment

The code has only been tested with the environment list below:
- python=3.8.13
- numpy=1.19.5
- torch=1.10.2+cu102
- tslearn=0.5.2
- scipy=1.6.2
- numba
- sktime

## Procedure

Please follow the steps below to reproduce the experiments.

1. download the UCR archive from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and unzip the dataset
2. change line 16 in ```./config/script/script_config_0.py``` to the path of the directory that contains the UCR archive
3. change the current folder to ```./config/script/```
4. run ```python script_config_0.py``` to generate all the config files
5. change the current folder back to ```.```
6. run the following commands to fine-tune/test models:
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode class_token``` for transformer + TimeCLR (the proposed method) with class token
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode flatten``` for transformer + TimeCLR (the proposed method) with flatten
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode gt``` for transformer + TimeCLR (the proposed method) with global token
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze false --aggregation_mode pooling --pooling_mode st``` for transformer + TimeCLR (the proposed method) with segment token
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode class_token``` for transformer + TimeCLR (the proposed method) with class token
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode flatten``` for transformer + TimeCLR (the proposed method) with flatten
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode gt``` for transformer + TimeCLR (the proposed method) with global token
    * ```python script_ucr_nn_0.py --method_name trf_tc_c_0000 --is_freeze true --aggregation_mode pooling --pooling_mode st``` for transformer + TimeCLR (the proposed method) with segment token
5. run ```python script_result_0.py``` to get the experiment results.


## Research

### Motivation

Time-series classification on foundation model
- Good performance in pre-trained domain, but model has difficulties in fine-tuning and overfitting
- Due to lack of time-series data
- Missing the temporal information and local pattern in classification

### Key Idea

Time-series data aggregation with temporal information
- ViT(Class token, Representation vector)
- Pooling(Static temporal pooling, Dynamic temporal pooling)
For high performance without large computations, the pre-trained backbone is frozen and fine-tuning is only applied to the downstream task

### Experiments Setting

- Data
  - UCR Time series Classification Archive
  - Labelled time-series data for a variety of sources including medical, financial, biological, industrial, and environmental

- Model Architecture
  - Foundation model are composed with backbone and head

  <center><img src="https://github.com/hataehyeok/time_seriesFM/assets/105369662/9e212373-d4a5-4d25-9bfa-9524fef23d02" width="90%" height="90%"/></center>
  
  - Transformer setting is fixed positional encoding, 4 encoder layer, 64 input size, output size and pre-trained by TimeCLR, which is contrastive learning pre-training method extends SimCLR.
  - Previous research use class token and ViT based time-series foundation model, which is trained when pre-training process
  - Output network, projector, classifier also trained on pre-training process

### Experiments

- Freeze Backbone
  - Use transformer pre-trained by TimeCLR

  <center><img src="https://github.com/hataehyeok/time_seriesFM/assets/105369662/485a6d99-c8f7-4583-9979-61893f0705db" width="60%" height="60%"/></center>

  - Already pre-trained and time-series is aggregated with class token, therefore, experiments setting has advantage for class token

### Conclusions

- In the freeze case, the flatten method was overfitting even though it had the most parameters, and this was improved by pooling
- Considering temporal information, class token and Static temporal pooling performed the best, which means these aggregate local pattern in time-series well
- As expected, removing the head and attaching the head to the downstream task yielded better results, and class token alone had more limitations, such as overfitting
- pre-trained head is not useful in downstream task because accuracy of separate pre-train head is more improved

### Future Work

- Experiment with other data sets that were not used in the pre-train to further study multi-domain and zero-shot reference
- Research aggregation techniques that better detect temporal information