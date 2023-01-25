#pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at 
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#  @@-COPYRIGHT-END-@@
# =============================================================================

# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

def load_pretrained_model(args, labels, label2id, id2label):

    # imagenet has too many classes to print
    logging.getLogger("transformers.configuration_utils").disabled = True

    model_type = args.model_type
    if model_type == "vit":
        from huggingface.baseline_models.vit.modeling_vit import ViTForImageClassification as Model
        from transformers import AutoConfig as Config
        from transformers import AutoFeatureExtractor as FeatureExtractor

        config = Config.from_pretrained(
            args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
        )
        config.return_dict = False
        feature_extractor = FeatureExtractor.from_pretrained(
            args.model_name_or_path,
        )
        
        interpolate = False
        if args.higher_resolution and feature_extractor.size == 224:
            feature_extractor.size = 384
            interpolate = True
            raise NotImplementedError("Need to quantize interpolate fn in modeling_vit.py")

        model = Model.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif model_type == "mobilevit":
        from huggingface.baseline_models.mobilevit.modeling_mobilevit import MobileViTForImageClassification as Model
        from transformers import MobileViTConfig as Config
        from transformers import MobileViTFeatureExtractor as FeatureExtractor
        
        config = Config.from_pretrained(args.model_name_or_path)
        config.return_dict = False
        model = Model.from_pretrained(args.model_name_or_path, config=config)
        feature_extractor = FeatureExtractor.from_pretrained(args.model_name_or_path)
        interpolate = False
    return model, feature_extractor, interpolate



