# -*- coding: utf-8 -*-
"""
@author: 
"""


import torch
from .classifier_0 import Classifier
from .time_freq_0 import TimeFreqEncoder
from .ts2vec_enc_0 import TS2VecEncoder
from .mixup_enc_0 import MixingUpEncoder
from .simclr_enc_0 import SimCLREncoder
from .timeclr_enc_0 import TimeCLREncoder
from .resnet1d_1 import ResNet1D as ResNet1D_1
from .rnnet_0 import RNNet
from .transformer_1 import Transformer

from ..augment import jittering
from ..augment import smoothing
from ..augment import mag_warping
from ..augment import add_slope
from ..augment import add_spike
from ..augment import add_step
from ..augment import cropping
from ..augment import masking
from ..augment import shifting
from ..augment import time_warping


def load_pretrain(model_config, encoder):
    if 'pre_train_model' not in model_config:
        return encoder

    pre_train_model = model_config['pre_train_model']
    pkl = torch.load(pre_train_model, map_location='cpu')
    # encoder.load_state_dict(pkl['model_state_dict'])
    encoder.load_state_dict(pkl['model_state_dict'], strict=False)
    return encoder


def get_rnnet(model_config):
    in_dim = int(model_config['encoder']['in_dim'])
    out_dim = int(model_config['encoder']['out_dim'])
    rnn_type = model_config['encoder']['rnn_type']
    n_layer = int(model_config['encoder']['n_layer'])
    n_dim = int(model_config['encoder']['n_dim'])
    is_projector = model_config['encoder']['is_projector']
    is_projector = is_projector.lower() == 'true'
    project_norm = model_config['encoder']['project_norm']
    dropout = float(model_config['encoder']['dropout'])

    encoder = RNNet(
        in_dim=in_dim, out_dim=out_dim, rnn_type=rnn_type,
        n_layer=n_layer, n_dim=n_dim, is_projector=is_projector,
        project_norm=project_norm, dropout=dropout)
    encoder = load_pretrain(model_config['encoder'], encoder)
    return encoder


def get_resnet1d(model_config):
    in_dim = int(model_config['encoder']['in_dim'])
    out_dim = int(model_config['encoder']['out_dim'])
    n_dim = int(model_config['encoder']['n_dim'])
    block_type = model_config['encoder']['block_type']
    norm = model_config['encoder']['norm']
    is_projector = model_config['encoder']['is_projector']
    is_projector = is_projector.lower() == 'true'
    project_norm = model_config['encoder']['project_norm']
    encoder = ResNet1D_1(
        in_dim=in_dim, out_dim=out_dim, n_dim=n_dim,
        block_type=block_type, norm=norm,
        is_projector=is_projector, project_norm=project_norm)

    encoder = load_pretrain(model_config['encoder'], encoder)
    return encoder


def get_transformer(model_config, aggregation_mode, pooling_mode):
    in_dim = int(model_config['encoder']['in_dim'])
    out_dim = int(model_config['encoder']['out_dim'])
    n_layer = int(model_config['encoder']['n_layer'])
    n_dim = int(model_config['encoder']['n_dim'])
    n_head = int(model_config['encoder']['n_head'])
    norm_first = model_config['encoder']['norm_first']
    norm_first = norm_first.lower() == 'true'
    is_pos = model_config['encoder']['is_pos']
    is_pos = is_pos.lower() == 'true'
    is_projector = model_config['encoder']['is_projector']
    is_projector = is_projector.lower() == 'true'
    project_norm = model_config['encoder']['project_norm']
    dropout = float(model_config['encoder']['dropout'])

    encoder = Transformer(
        in_dim=in_dim, out_dim=out_dim, n_layer=n_layer,
        n_dim=n_dim, n_head=n_head, norm_first=norm_first,
        is_pos=is_pos, is_projector=is_projector,
        project_norm=project_norm, dropout=dropout, aggregation_mode=aggregation_mode, pooling_mode=pooling_mode)
    encoder = load_pretrain(model_config['encoder'], encoder)
    return encoder


def get_timefreq(model_config, encoder):
    jitter_strength = float(model_config['timefreq']['jitter_strength'])
    freq_ratio = float(model_config['timefreq']['freq_ratio'])
    freq_strength = float(model_config['timefreq']['freq_strength'])
    project_norm = model_config['timefreq']['project_norm']
    encoder_ = TimeFreqEncoder(
        encoder, jitter_strength=jitter_strength, freq_ratio=freq_ratio,
        freq_strength=freq_strength, project_norm=project_norm)
    encoder_ = load_pretrain(model_config['timefreq'], encoder_)
    return encoder_


def get_ts2vec(model_config, encoder):
    encoder_ = TS2VecEncoder(encoder)
    encoder_ = load_pretrain(model_config['ts2vec'], encoder_)
    return encoder_


def get_mixup(model_config, encoder):
    encoder_ = MixingUpEncoder(encoder)
    encoder_ = load_pretrain(model_config['mixup'], encoder_)
    return encoder_


def get_simclr(model_config, encoder):
    encoder_ = SimCLREncoder(encoder)
    encoder_ = load_pretrain(model_config['simclr'], encoder_)
    return encoder_



def get_timeclr(model_config, encoder):
    aug_bank_ver = int(model_config['timeclr']['aug_bank_ver'])
    if aug_bank_ver == 0:
        aug_bank = [
            lambda x:jittering(x, strength=0.1, seed=None),
            lambda x:smoothing(x, max_ratio=0.5, min_ratio=0.01, seed=None),
            lambda x:mag_warping(x, strength=1, seed=None),
            lambda x:add_slope(x, strength=1, seed=None),
            lambda x:add_spike(x, strength=3, seed=None),
            lambda x:add_step(x, min_ratio=0.1, strength=1, seed=None),
            lambda x:cropping(x, min_ratio=0.1, seed=None),
            lambda x:masking(x, max_ratio=0.5, seed=None),
            lambda x:shifting(x, seed=None),
            lambda x:time_warping(x, min_ratio=0.5, seed=None),
        ]

    encoder_ = TimeCLREncoder(encoder, aug_bank)
    encoder_ = load_pretrain(model_config['timeclr'], encoder_)
    return encoder_


def get_classifier(model_config, encoder):
    n_class = int(model_config['classifier']['n_class'])
    n_dim = int(model_config['classifier']['n_dim'])
    n_layer = int(model_config['classifier']['n_layer'])
    model = Classifier(
        encoder, n_class, n_dim=n_dim, n_layer=n_layer)
    return model


def get_model(model_config, aggregation_mode, pooling_mode):
    model_name = model_config['model']['model_name']
    print(f'get model for {model_name}')

    if model_config['encoder']['in_dim'] == None:
        model_config['encoder']['in_dim'] = model_config['in_dim']

    if 'rnnet' in model_name:
        print('  get rnnet')
        encoder = get_rnnet(model_config)
    elif 'resnet1d' in model_name:
        print('  get resnet1d')
        encoder = get_resnet1d(model_config)
    elif 'transform' in model_name:
        print('  get transform')
        encoder = get_transformer(model_config, aggregation_mode, pooling_mode)
    else:
        raise Exception(f'unknown encoder name: {model_name}')

    if 'timefreq' in model_name:
        print('  get timefreq')
        encoder = get_timefreq(model_config, encoder)
    elif 'ts2vec' in model_name:
        print('  get ts2vec')
        encoder = get_ts2vec(model_config, encoder)
    elif 'mixup' in model_name:
        print('  get mixup')
        encoder = get_mixup(model_config, encoder)
    elif 'simclr' in model_name:
        print('  get simclr')
        encoder = get_simclr(model_config, encoder)
    elif 'timeclr' in model_name:
        print('  get timeclr')
        encoder = get_timeclr(model_config, encoder)

    if 'classifier' in model_name:
        print('  get classifier')
        model_config['classifier']['n_class'] = model_config['n_class']
        model = get_classifier(model_config, encoder)
    else:
        model = encoder
    return model

