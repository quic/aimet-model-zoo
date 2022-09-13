#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


def get_tar_path(model_index, model_spec_index):

    # ABPN
    if model_index == 0:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/abpn-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_abpn_28_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_abpn_28_3x.tar.gz'
        elif model_spec_index == 2:
            tar_path += 'release_abpn_28_4x.tar.gz'
        elif model_spec_index == 3:
            tar_path += 'release_abpn_32_2x.tar.gz'
        elif model_spec_index == 4:
            tar_path += 'release_abpn_32_3x.tar.gz'
        else:
            tar_path += 'release_abpn_32_4x.tar.gz'

    # XLSR
    elif model_index == 1:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/xlsr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_xlsr_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_xlsr_3x.tar.gz'
        else:
            tar_path += 'release_xlsr_4x.tar.gz'

    # SESR
    elif model_index == 2:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/sesr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_sesr_m3_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_sesr_m3_3x.tar.gz'
        else:
            tar_path += 'release_sesr_m3_4x.tar.gz'
    elif model_index == 3:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/sesr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_sesr_m5_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_sesr_m5_3x.tar.gz'
        else:
            tar_path += 'release_sesr_m5_4x.tar.gz'
    elif model_index == 4:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/sesr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_sesr_m7_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_sesr_m7_3x.tar.gz'
        else:
            tar_path += 'release_sesr_m7_4x.tar.gz'
    elif model_index == 5:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/sesr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_sesr_m11_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_sesr_m11_3x.tar.gz'
        else:
            tar_path += 'release_sesr_m11_4x.tar.gz'
    elif model_index == 6:
        tar_path = 'https://github.com/quic/aimet-model-zoo/releases/download/sesr-checkpoint-pytorch/'
        if model_spec_index == 0:
            tar_path += 'release_sesr_xl_2x.tar.gz'
        elif model_spec_index == 1:
            tar_path += 'release_sesr_xl_3x.tar.gz'
        else:
            tar_path += 'release_sesr_xl_4x.tar.gz'

    return tar_path