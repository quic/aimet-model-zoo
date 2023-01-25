#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" quantization evaluation script of Bert-like models  (Bert, DistilBert, MiniLM, Roberta, MobileBert) for GLUE dataset
"""

# coding=utf-8
# python import
import logging
import os
import urllib
import progressbar
import numpy as np
import torch


# transformers import
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# AIMET imports
from aimet_torch.quantsim import load_checkpoint

from datasets import load_dataset, load_metric

# Utils imports
from utils.utils_nlclassifier_dataclass import (
    ModelArguments,
    DataTrainingArguments,
    AuxArguments,
)

# To be replaced once released
OFFICIAL_URL_HEAD = "https://github.com/quic/aimet-model-zoo/releases/download/torch_distilbert"

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

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


class DownloadProgressBar:
    """Download progress bar
    """
    def __init__(self):
        self.dpb = None

    def __call__(self, b_num, b_size, size):
        widgets = [
            "\x1b[33mDownloading weights \x1b[39m",
            progressbar.Percentage(),
            progressbar.Bar(marker="\x1b[32m#\x1b[39m"),
        ]
        if self.dpb is  None:
            self.dpb = progressbar.ProgressBar(
                widgets=widgets, maxval=size, redirect_stdout=True
            )
            self.dpb.start()

        processed = b_num * b_size
        if processed >= size:
            self.dpb.finish()
        else:
            self.dpb.update(processed)


def download_weights(data_args):
    """Download weights to cache directory
    """
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    url_checkpoint_test = f"{OFFICIAL_URL_HEAD}/{data_args.task_name}_fp.pth"
    urllib.request.urlretrieve(
        url_checkpoint_test, "./.cache/fp.pth", DownloadProgressBar()
    )
    url_checkpoint_test = f"{OFFICIAL_URL_HEAD}/{data_args.task_name}_qat.ckpt"
    urllib.request.urlretrieve(
        url_checkpoint_test, "./.cache/qat.ckpt", DownloadProgressBar()
    )


def main():
    """main function for quantization evaluation
    """
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         TrainingArguments,
         AuxArguments))
    (
        model_args,
        data_args,
        training_args,
        aux_args,
    ) = parser.parse_args_into_dataclasses()

    # download weights of original and quantized weight files

    print("===========download weights====================")
    download_weights(data_args) 

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    # Set the verbosity to info of the Transformers logger (on main process
    # only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files:
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your
        # needs.
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            label_list = sorted(datasets["train"].unique("label"))
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # ++++
    config.return_dict = False
    config.classifier_dropout = None
    config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob

    # ++++
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = torch.load(aux_args.fmodel_path)

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak
        # to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if (
                "sentence1" in non_label_column_names and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence
        # length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure
    # we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k,
            v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "+
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."+
                "\nIgnoring the model labels as a result."
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        """Tokenize the texts
        """
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args,
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets[
        "validation_matched" if data_args.task_name == "mnli" else "validation"
    ]

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)


    def compute_metrics(p: EvalPrediction):
        """ computer metrics
        You can define your custom compute_metrics function. It takes an `EvalPrediction`
        object (a namedtuple with a predictions and label_ids field) and has to return
        a dictionary string to float
        """
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(
                    list(result.values())).item()
            return result
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        return {
            "accuracy": (
                preds == p.label_ids).astype(
                    np.float32).mean().item()}

    # Load the Quantsim_modl object
    quantsim_model = load_checkpoint(aux_args.qmodel_path)

    # Initialize Trainer for evaluation
    model.cuda()
    model.eval()
    quantsim_model.model.cuda()
    quantsim_model.model.eval()
    ftrainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )
    qtrainer = Trainer(
        model=quantsim_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Evaluation
    feval_results, qeval_results = {}, {}
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    for eval_dataset, _ in zip(eval_datasets, tasks):
        feval_result = ftrainer.evaluate(eval_dataset=eval_dataset)
        qeval_result = qtrainer.evaluate(eval_dataset=eval_dataset)
        # FP
        foutput_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_fp.txt"
        )
        if ftrainer.is_world_process_zero():
            with open(foutput_eval_file, "w") as writer:
                logger.info(f"***** FP Eval results *****")
                for key, value in sorted(feval_result.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                writer.write(
                    f"Memory usage:\n{torch.cuda.memory_summary(device=0, abbreviated=False)}"
                )
        feval_results.update(feval_result)
        # QAT
        qoutput_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_qat.txt"
        )
        if qtrainer.is_world_process_zero():
            with open(qoutput_eval_file, "w") as writer:
                logger.info(f"***** QAT Eval results *****")
                for key, value in sorted(qeval_result.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                writer.write(
                    f"Memory usage:\n{torch.cuda.memory_summary(device=0, abbreviated=False)}"
                )
        qeval_results.update(qeval_result)


if __name__ == "__main__":
    main()
