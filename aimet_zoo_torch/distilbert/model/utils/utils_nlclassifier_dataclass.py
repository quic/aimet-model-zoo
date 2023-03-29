# /usr/bin/env python3
#pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at 
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


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


from dataclasses import dataclass,field 
from typing import Optional

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`."},
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None ,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`."},
    )
    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "probability of atention"}
    )

@dataclass
class AuxArguments:
    """
    Auxiliary arguments pertaining to training.
    """
    fmodel_path: str = field(
        default="../model/weights/fp.pth",
        metadata={"help": "Path to the full-precision model"}
    )
    qmodel_path: str = field(
        default="../model/weights/qat.ckpt",
        metadata={"help": "Path to the quantized model"}
    )
    model_config: str = field(
        default="bert_w8a8_rte",
        metadata={"help": "Model configuration"}
    )
