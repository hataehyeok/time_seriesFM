# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import numpy as np
from scipy import signal
from .util import _relabel
from .util import _normalize_dataset


def load_ucr_dataset(data_name, data_config):
    data_config = data_config['data']
    data_dir = data_config['data_dir']
    max_len = int(data_config['max_len'])
    seed = int(data_config['seed'])
    pretrain_frac = float(data_config['pretrain_frac'])
    train_frac = float(data_config['train_frac'])
    valid_frac = float(data_config['valid_frac'])
    test_frac = float(data_config['test_frac'])
    is_same_length = data_config['is_same_length']
    is_same_length = is_same_length.lower() == 'true'
    # assert pretrain_frac + train_frac + valid_frac + test_frac == 1.0

    reduced_train = 0
    if 'reduced_train' in data_config:
        reduced_train = float(data_config['reduced_train'])

    train_path = os.path.join(
        data_dir, 'Missing_value_and_variable_length_datasets_adjusted',
        '{0}', '{0}_TRAIN.tsv')
    test_path = os.path.join(
        data_dir, 'Missing_value_and_variable_length_datasets_adjusted',
        '{0}', '{0}_TEST.tsv')
    if not os.path.isfile(train_path.format(data_name)):
        train_path = os.path.join(data_dir, '{0}', '{0}_TRAIN.tsv')
        test_path = os.path.join(data_dir, '{0}', '{0}_TEST.tsv')

    train_path = train_path.format(data_name)
    test_path = test_path.format(data_name)

    data = np.concatenate(
        (np.loadtxt(train_path),
         np.loadtxt(test_path), ), axis=0)
    n_data = data.shape[0]
    data_len = data.shape[1] - 1

    np.random.seed(seed=seed)
    random_vec = np.random.permutation(n_data)
    data = data[random_vec, :]

    label = data[:, 0]
    label = label.astype(int)
    data = data[:, 1:]
    data = np.expand_dims(data, 1)

    if is_same_length:
        if data_len != max_len:
            data = signal.resample(
                data, max_len, axis=2)
    else:
        if data_len > max_len:
            data = signal.resample(
                data, max_len, axis=2)

    label, n_class = _relabel(label)
    if np.isclose(pretrain_frac, 1.0):
        data_pretrain = data
        data_train = None
        data_valid = None
        data_test = None
        label_train = None
        label_valid = None
        label_test = None
    else:
        data_pretrain = []
        data_train = []
        data_valid = []
        data_test = []
        label_train = []
        label_valid = []
        label_test = []
        for i in range(n_class):
            data_i = data[label == i, :, :]
            label_i = label[label == i]

            n_data_i = label_i.shape[0]
            n_train_i = np.round(train_frac * n_data_i)
            n_train_i = int(n_train_i)
            n_train_i = max(n_train_i, 1)

            n_valid_i = np.round(valid_frac * n_data_i)
            n_valid_i = int(n_valid_i)
            n_valid_i = max(n_valid_i, 1)

            n_test_i = np.round(test_frac * n_data_i)
            n_test_i = int(n_test_i)
            n_test_i = max(n_test_i, 1)

            n_pretrain_i = n_data_i - n_train_i - n_valid_i - n_test_i

            train_start = 0
            train_end = n_train_i

            valid_start = train_end
            valid_end = valid_start + n_valid_i

            test_start = valid_end
            test_end = test_start + n_test_i

            pretrain_start = test_end
            pretrain_end = pretrain_start + n_pretrain_i

            if reduced_train > 0:
                train_end *= reduced_train
                train_end = np.round(train_end)
                if train_end < 1:
                    train_end = 1
                train_end = int(train_end)

            data_train.append(data_i[train_start:train_end, :, :])
            data_valid.append(data_i[valid_start:valid_end, :, :])
            data_test.append(data_i[test_start:test_end, :, :])
            data_pretrain.append(data_i[pretrain_start:pretrain_end, :, :])

            label_train.append(label_i[train_start:train_end])
            label_valid.append(label_i[valid_start:valid_end])
            label_test.append(label_i[test_start:test_end])

        data_train = np.concatenate(data_train, axis=0)
        data_valid = np.concatenate(data_valid, axis=0)
        data_test = np.concatenate(data_test, axis=0)
        data_pretrain = np.concatenate(data_pretrain, axis=0)
        label_train = np.concatenate(label_train, axis=0)
        label_valid = np.concatenate(label_valid, axis=0)
        label_test = np.concatenate(label_test, axis=0)

    dataset_ = {}
    dataset_['data_pretrain'] = data_pretrain
    dataset_['data_train'] = data_train
    dataset_['data_valid'] = data_valid
    dataset_['data_test'] = data_test
    dataset_['label_train'] = label_train
    dataset_['label_valid'] = label_valid
    dataset_['label_test'] = label_test
    dataset_['n_class'] = n_class
    dataset_['n_dim'] = data_pretrain.shape[1]
    dataset_['data_len'] = data_pretrain.shape[2]
    return dataset_


def load_ucr_pretrain(data_config):
    data_names = get_ucr_data_names()
    pretrain_data = []
    max_len = 0
    for data_name in data_names:
        dataset = load_ucr_dataset(data_name, data_config)
        data_pretrain = _normalize_dataset(dataset['data_pretrain'])
        pretrain_data.append(data_pretrain)
        if max_len < dataset['data_len']:
            max_len = dataset['data_len']

    for i in range(len(pretrain_data)):
        data_len = pretrain_data[i].shape[2]
        if data_len < max_len:
            n_data = pretrain_data[i].shape[0]
            data_i = np.zeros((n_data, 1, max_len))
            data_i[:, :, :data_len] = pretrain_data[i]
            pretrain_data[i] = data_i
    pretrain_data = np.concatenate(pretrain_data, axis=0)
    return pretrain_data


def get_ucr_data_names():
    names = [
        'Adiac',
        'ArrowHead',
        'Beef',
        'BeetleFly',
        'BirdChicken',
        'Car',
        'CBF',
        'ChlorineConcentration',
        'CinCECGTorso',
        'Coffee',
        'Computers',
        'CricketX',
        'CricketY',
        'CricketZ',
        'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup',
        'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW',
        'Earthquakes',
        'ECG200',
        'ECG5000',
        'ECGFiveDays',
        'ElectricDevices',
        'FaceAll',
        'FaceFour',
        'FacesUCR',
        'FiftyWords',
        'Fish',
        'FordA',
        'FordB',
        'GunPoint',
        'Ham',
        'HandOutlines',
        'Haptics',
        'Herring',
        'InlineSkate',
        'InsectWingbeatSound',
        'ItalyPowerDemand',
        'LargeKitchenAppliances',
        'Lightning2',
        'Lightning7',
        'Mallat',
        'Meat',
        'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup',
        'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW',
        'MoteStrain',
        'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2',
        'OliveOil',
        'OSULeaf',
        'PhalangesOutlinesCorrect',
        'Phoneme',
        'Plane',
        'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect',
        'ProximalPhalanxTW',
        'RefrigerationDevices',
        'ScreenType',
        'ShapeletSim',
        'ShapesAll',
        'SmallKitchenAppliances',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'StarLightCurves',
        'Strawberry',
        'SwedishLeaf',
        'Symbols',
        'SyntheticControl',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'TwoLeadECG',
        'TwoPatterns',
        'UWaveGestureLibraryAll',
        'UWaveGestureLibraryX',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'Wafer',
        'Wine',
        'WordSynonyms',
        'Worms',
        'WormsTwoClass',
        'Yoga',
        'ACSF1',
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'DodgerLoopDay',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'EthanolLevel',
        'FreezerRegularTrain',
        'FreezerSmallTrain',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'MixedShapesRegularTrain',
        'MixedShapesSmallTrain',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD',
    ]
    return names

