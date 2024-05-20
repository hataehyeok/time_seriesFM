# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import pathlib
from engine import get_dataset
from engine import get_dist_classifier
from engine import get_base_classifier
from engine import get_pretrain_model


def main():
    ucr_dir = '/home/hth021002/ai_research/UCRArchive_2018'

    config_dir = os.path.join('..', 'file')
    path = pathlib.Path(config_dir)
    path.mkdir(parents=True, exist_ok=True)

    get_dataset(ucr_dir)
    get_dist_classifier()
    get_base_classifier()
    get_pretrain_model()


if __name__ == '__main__':
    main()

