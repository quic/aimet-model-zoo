# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R0903
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""AIMET Quantization evaluation code of MobileVIT for image classification"""


import argparse
import logging
import transformers
from transformers.utils.versions import require_version
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from aimet_zoo_torch.mobilevit.dataloader import get_dataloaders
from aimet_zoo_torch.mobilevit import mobilevit

logger = get_logger(__name__)


require_version(
    "datasets>=2.0.0",
    "To fix: pip install datasets==2.4.0",
)


def parse_args(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluating VIT/MobileVIT Transformers model on an imagenet dataset"
    )
    parser.add_argument(
        "--model_config",
        default="mobilevit_w8a8",
        help="choice [mobilevit_w8a8]",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default=None,
        help="A folder containing the validation data.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        help="training seed",
    )    
    args = parser.parse_args(raw_args)

    return args


def main(raw_args=None):
    """Evaluation main function"""
    args = parse_args(raw_args)
    # Initialize the accelerator. We will let the accelerator
    # handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here
    # and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator()
    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # loading finetuned original model
    model = mobilevit(model_config=args.model_config, quantized=False)
    model_orig = model.get_model_from_pretrained()

    feature_extractor = model.get_feature_extractor_from_pretrained()

    # load modularized eval_function and dataloaders
    train_dataloader, eval_dataloader, eval_function = get_dataloaders(
        args, feature_extractor
    )

    # Prepare everything with our `accelerator`.
    model_orig, train_dataloader, eval_dataloader = accelerator.prepare(
        model_orig, train_dataloader, eval_dataloader
    )

    iterations = 1e5
    metric = datasets.load_metric("accuracy")
    # original model performance
    original_model_performance_fp32 = eval_function(
        model_orig, [iterations, eval_dataloader, metric]
    )

    # get quantsim for original model
    sim_orig = model.get_quantsim(train_dataloader, eval_dataloader, eval_function)

    # original model performance on 8bit device
    original_model_performance_int8 = eval_function(
        sim_orig.model, [iterations, eval_dataloader, metric]
    )
    del model

    # loading optimized model
    model = mobilevit(model_config=args.model_config, quantized=True)
    model_w8a8 = model.get_model_from_pretrained()

    # Prepare everything with our `accelerator`.
    model_w8a8, train_dataloader, eval_dataloader = accelerator.prepare(
        model_w8a8, train_dataloader, eval_dataloader
    )
    # optimized model on 32bit device
    quantized_model_performance_fp32 = eval_function(
        model_w8a8, [iterations, eval_dataloader, metric]
    )
    # get quantsim for optimized model
    sim_w8a8 = model.get_quantsim(train_dataloader, eval_dataloader, eval_function)
    # optimized model performance on 8bit device
    quantized_model_performance_int8 = eval_function(
        sim_w8a8.model, [iterations, eval_dataloader, metric]
    )

    logger.info(f"Original model performances")
    logger.info(f"===========================")
    logger.info(
        f"Original Model | 32-bit Environment | perplexity : {original_model_performance_fp32:.4f}"
    )
    logger.info(
        f"Original Model |  8-bit Environment | perplexity: {original_model_performance_int8:.4f}"
    )
    logger.info(f"Optimized model performances")
    logger.info(f"===========================")
    logger.info(
        f"Optimized Model | 32-bit Environment | perplexity: {quantized_model_performance_fp32:.4f}"
    )
    logger.info(
        f"Optimized Model |  8-bit Environment | perplexity: {quantized_model_performance_int8:.4f}"
    )
    
    return {
        'original_model_performance_fp32':original_model_performance_fp32,
        'original_model_performance_int8':original_model_performance_int8,
        'quantized_model_performance_fp32':quantized_model_performance_fp32,
        'quantized_model_performance_int8':quantized_model_performance_int8
        }

if __name__ == "__main__":
    main()
