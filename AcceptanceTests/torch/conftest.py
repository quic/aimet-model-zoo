# /usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================

"""pytest configuration fixtures"""

import pytest
import os
import warnings
from pathlib import Path


@pytest.fixture(scope='module')
def test_data_path():
    """
    This fixture will return the path to testing data for all models. When no
    DEPENDENCY_DATA_PATH is detected, None is returned and all acceptance tests
    which depend on testing data will be xfailed
    """
    try:
        data_path = Path(os.getenv('DEPENDENCY_DATA_PATH'))
    except TypeError:
        warnings.warn('In order to successfully proceed with acceptance test, please set DEPENDENCY_DATA_PATH')
        data_path = None

    yield data_path

@pytest.fixture(scope='module')
def tiny_imageNet_root_path(test_data_path):
    if test_data_path is not None:
        tiny_imageNet_root_path = (test_data_path / 'model_zoo_datasets/ILSVRC2012_PyTorch_reduced').as_posix() 
    else:
        tiny_imageNet_root_path = None

    yield tiny_imageNet_root_path


@pytest.fixture(scope='module')
def tiny_imageNet_validation_path(test_data_path):
    if test_data_path is not None:
        tiny_imageNet_validation_path = (test_data_path / 'model_zoo_datasets/ILSVRC2012_PyTorch_reduced/val').as_posix() 
    else:
        tiny_imageNet_validation_path = None

    yield tiny_imageNet_validation_path

@pytest.fixture(scope='module')
def tiny_imageNet_train_path(test_data_path):
    if test_data_path is not None:
        tiny_imageNet_train_path = (test_data_path / 'model_zoo_datasets/ILSVRC2012_PyTorch_reduced/train').as_posix() 
    else:
        tiny_imageNet_train_path = None

    yield tiny_imageNet_train_path


@pytest.fixture(scope='module')
def tiny_mscoco_validation_path(test_data_path):
    if test_data_path is not None:
        tiny_mscoco_validation_path = (test_data_path / "model_zoo_datasets/tiny_coco/val_2017").as_posix() 
    else:
        tiny_mscoco_validation_path = None

    yield tiny_mscoco_validation_path.as_posix()


@pytest.fixture(scope='module')
def tiny_cityscapes_path(test_data_path):
    if test_data_path is not None:
        tiny_cityscapes_path = (test_data_path / "model_zoo_datasets/tiny_cityscapes").as_posix() 
    else:
        tiny_cityscapes_path = None

    yield tiny_cityscapes_path


@pytest.fixture(scope='module')
def super_resolution_set5_path(test_data_path):
    if test_data_path is not None:
        super_resolution_set5_path = (test_data_path / "model_zoo_datasets/super_resolution_data/Set5/image_SRF_4_HR").as_posix() 
    else:
        super_resolution_set5_path = None

    yield super_resolution_set5_path


@pytest.fixture(scope='module')
def PascalVOC_segmentation_test_data_path(test_data_path):
    if test_data_path is not None:
        pascalVOC_segmentation_path = (test_data_path / 'PascalVOCSegmentation').as_posix() 
    else:
        pascalVOC_segmentation_path = None

    yield pascalVOC_segmentation_path
