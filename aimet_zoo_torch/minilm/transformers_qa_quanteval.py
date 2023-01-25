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
""" quantization evaluation script of Bert-like models  (Bert, DistilBert, MiniLM, Roberta, MobileBert) for SQUAD dataset

"""
import logging
import os
import sys
import urllib
import torch
import progressbar


from aimet_torch.quantsim import load_checkpoint
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from aimet_zoo_torch.bert.utils.utils_qa_dataclass import (
    ModelArguments,
    DataTrainingArguments,
    AuxArguments,
)

from datasets import load_dataset, load_metric
import datasets

from utils.utils_qa import postprocess_qa_predictions
from utils.trainer_qa import QuestionAnsweringTrainer


OFFICIAL_URL_HEAD = "https://github.com/quic/aimet-model-zoo/releases/download/torch_minilm"

# Utils imports

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


class DownloadProgressBar:
    """Download progress bar for downloading
    """
    def __init__(self):
        self.dpb = None

    def __call__(self, b_num, b_size, size):
        widgets = [
            "\x1b[33mDownloading weights \x1b[39m",
            progressbar.Percentage(),
            progressbar.Bar(marker="\x1b[32m#\x1b[39m"),
        ]
        if self.dpb is None:
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

    Args:
        data_args (_type_): arguments for dataset
    """
    # Download weights to cache directory
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    url_checkpoint_test = f"{OFFICIAL_URL_HEAD}/{data_args.dataset_name}_fp.pth"
    urllib.request.urlretrieve(
        url_checkpoint_test, "./.cache/fp.pth", DownloadProgressBar()
    )
    url_checkpoint_test = f"{OFFICIAL_URL_HEAD}/{data_args.dataset_name}_qat.ckpt"
    urllib.request.urlretrieve(
        url_checkpoint_test, "./.cache/qat.ckpt", DownloadProgressBar()
    )


def main():
    """ main evaluation script
    """
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         TrainingArguments,
         AuxArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, aux_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            aux_args,
        ) = parser.parse_args_into_dataclasses()

    # ++++hardcoded values
    training_args.overwrite_output_dir = True
    training_args.do_eval = True
    # download weights of original and quantized weight files
    print("===========download weights====================")
    download_weights(data_args) 

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # ++++
    config.return_dict = False
    config.classifier_dropout = None
    # ++++
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = torch.load(aux_args.fmodel_path)

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or
    # (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        """ validation preprocessing
        """
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is
            # the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the
            # example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(
                range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(
                desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will
            # select required samples again
            eval_dataset = eval_dataset.select(
                range(data_args.max_eval_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(
            examples,
            features,
            predictions,
            stage="eval"):
        """Post-processing: we match the start logits and end logits to answers in the original context.
        """

        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                      for ex in examples]
        return EvalPrediction(
            predictions=formatted_predictions,
            label_ids=references)

    # metric function
    metric = load_metric(
        "squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        """  calculate compute metrics using EvalPrediction

        """
        return metric.compute(
            predictions=p.predictions,
            references=p.label_ids)

    # Load the Quantsim_model object
    quantsim_model = load_checkpoint(aux_args.qmodel_path)

    # Initialize Trainer for evaluation
    model.cuda()
    model.eval()
    quantsim_model.model.cuda()
    quantsim_model.model.eval()

    ftrainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    qtrainer = QuestionAnsweringTrainer(
        model=quantsim_model.model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** FP32 Evaluate ***")
    metrics = ftrainer.evaluate()

    max_eval_samples = (
        data_args.max_eval_samples
        if data_args.max_eval_samples is not None
        else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    ftrainer.log_metrics("eval", metrics)
    ftrainer.save_metrics("eval", metrics)

    logger.info("*** Quantized model Evaluate ***")
    metrics = qtrainer.evaluate()

    max_eval_samples = (
        data_args.max_eval_samples
        if data_args.max_eval_samples is not None
        else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    qtrainer.log_metrics("eval", metrics)
    qtrainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
