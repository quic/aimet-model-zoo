# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch

class NoScalingScaler:
    state_dict_key = "amp_scaler" #"noscaling_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        optimizer.step()


    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
