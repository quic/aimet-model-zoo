# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" utility functions for loading encoding, getting dummy input or get quantsim object"""

import math
import os
import csv

import numpy as np
import torch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_common.defs import QuantScheme


def get_dummy_input(loader):
    for batch in loader:
        output = []
        input_args = ["input_ids"]

        for k in input_args:
            if k in batch.keys():
                output.append(batch[k].to("cuda"))
            else:
                raise ValueError("dummy data error")
        return tuple(output)


def evaluate_model(model, iterations, loader, metric):
    model.eval()
    losses = []
    for step, batch in enumerate(loader):
        if step < iterations:
            for k in batch.keys():
                batch[k] = batch[k].to("cuda")
            with torch.no_grad():
                outputs = model(**batch)
            losses.append(outputs[0].item())
        else:
            break
    loss = np.mean(losses)
    if metric == "loss":
        return loss
    elif metric == "perplexity":
        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return perplexity
    else:
        raise ValueError("invalid metric: ", metric)


def eval_wrapper(model, args):
    iterations, loader, metric = args
    return evaluate_model(model, iterations, loader, metric)


def clamp_quantizer(quantized_model, clamp_min, clamp_max):
    for name, param in quantized_model.named_parameters():
        if name.endswith("encoding_min") or name.endswith("encoding_max"):
            if param.data.item() > clamp_max:
                print(
                    f"param {name} will be clipped from {param.data.item()} to {clamp_max}"
                )
                param.data = torch.Tensor([clamp_max]).to(param.device)
            elif param.data.item() < clamp_min:
                print(
                    f"param {name} will be clipped from {param.data.item()} to {clamp_min}"
                )
                param.data = torch.Tensor([clamp_min]).to(param.device)


def quantize_model(model, train_dataloader, eval_dataloader, config):

    metric = "perplexity"
    model.eval()

    dummy_input = get_dummy_input(train_dataloader)
    if config.quant_scheme == "tf":
        quant_scheme = QuantScheme.post_training_tf
    elif config.quant_scheme == "tf_enhanced":
        quant_scheme = QuantScheme.post_training_tf_enhanced
    elif config.quant_scheme == "tf_range_learning":
        quant_scheme = QuantScheme.training_range_learning_with_tf_init
    else:
        raise ValueError(
            "select appropriate quantization scheme in [tf, tf_enhanced, tf_range_learning]"
        )

    full_precision_model_performance = evaluate_model(
        model, 1e5, eval_dataloader, metric
    )

    quant_sim = QuantizationSimModel(
        model=model,
        quant_scheme=quant_scheme,
        dummy_input=dummy_input,
        rounding_mode="nearest",
        default_output_bw=config.activation_bit_width,
        default_param_bw=config.parameter_bit_width,
        in_place=True,
        config_file=config.config_file,
    )

    # remove dropout quantizers
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(
            module._module_to_wrap, torch.nn.Dropout
        ):
            module.output_quantizers[0].enabled = False

    quant_sim.compute_encodings(eval_wrapper, (10, eval_dataloader, metric))
    if config.clamp_quantizer:
        assert config.quant_scheme == "tf_range_learning"
        clamp_quantizer(quant_sim.model, -config.clamping_value, config.clamping_value)

    # load encodings if there is encodings.csv
    load_encoding_data(quant_sim, config.model_name_or_path)

    iterations=1e5
    quantized_model_performance = evaluate_model(
        quant_sim.model, iterations, eval_dataloader, metric
    )
    return quant_sim, full_precision_model_performance, quantized_model_performance


def load_encoding_data(quant_sim, save_dir):
    fname = os.path.join(save_dir, "encodings.csv")
    if not os.path.exists(fname):
        return

    print(f"loading encoding data from {fname}")

    def _load_data(fname):
        datadict = {}
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                datadict[row[0]] = float(row[1])
        return datadict

    enc = _load_data(fname)
    for name, param in quant_sim.model.named_parameters():
        if name.endswith("encoding_min") or name.endswith("encoding_max"):
            if name not in enc:
                print(
                    f"{name} is not in the pretrained encodings! TF intiailization will be used"
                )
            else:
                param.data = torch.Tensor([enc[name]]).to(param.device)
