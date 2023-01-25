#pylint: skip-file

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


from tqdm import tqdm
import torch
import datasets
import os
import copy
import csv

from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_torch.qc_quantize_op import QcQuantizeWrapper

def get_dummy_input(loader):
    for batch in loader:
        output = []
        input_args = ["pixel_values"]

        for k in input_args:
            if k in batch.keys():
                output.append(batch[k].to("cuda"))
        return tuple(output)

def evaluate_model(model, iterations, loader, metric):
    losses = []
    for step, batch in enumerate(tqdm(loader)):
        if step < iterations:
            for k in batch.keys():
                if k != "interpolate_pos_encoding":
                    batch[k] = batch[k].to('cuda')
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs[1].argmax(dim=-1)
            
            metric.add_batch(
                predictions=predictions,
                references=batch['labels'],
            )
        else:
            break
    return metric.compute()["accuracy"]

def eval_wrapper(model, args):
    iterations, loader, metric = args
    return evaluate_model(model, iterations, loader, metric)


def quantize_model(model, train_dataloader, eval_dataloader, tokenizer, config):

    metric = datasets.load_metric("accuracy")
    model.eval()
    
    full_precision_model_performance = evaluate_model(model, 1e+5, eval_dataloader, metric)

    dummy_input = get_dummy_input(train_dataloader)

    if config.quant_scheme == "tf":
        quant_scheme = QuantScheme.post_training_tf
    elif config.quant_scheme == 'tf_enhanced':
        quant_scheme = QuantScheme.post_training_tf_enhanced
    elif config.quant_scheme == 'tf_range_learning':
        quant_scheme = QuantScheme.training_range_learning_with_tf_init
    else:
        raise ValueError("select appropriate quantization scheme in [tf, tf_enhanced, tf_range_learning]")

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

    quant_sim.compute_encodings(eval_wrapper, (10, eval_dataloader, metric))
    if config.clamp_quantizer:
        assert config.quant_scheme == "tf_range_learning"
        clamp_quantizer(quant_sim.model, -30., 30.)
    
    # load encodings if there is encodings.csv
    load_encoding_data(quant_sim, config.model_name_or_path)


    # remove dropout quantizers
    disable_list = []
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.Dropout):
            disable_list.append(module)
    for module in disable_list:
        module.output_quantizers[0].enabled = False

    quantized_model_performance = evaluate_model(quant_sim.model, 1e+5, eval_dataloader, metric)
    return quant_sim, full_precision_model_performance, quantized_model_performance


def load_encoding_data(quant_sim, save_dir):
    fname = os.path.join(save_dir, "encodings.csv")
    if not os.path.exists(fname):
        return

    def _load_data(fname):
        datadict = {}
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                datadict[row[0]] = float(row[1])
        return datadict
    
    enc = _load_data(fname)
    for name, param in quant_sim.model.named_parameters():
        if name.endswith("encoding_min") or name.endswith("encoding_max"):
            param.data = torch.Tensor([enc[name]]).to(param.device)


def save_quantized_model(quant_sim, output_dir):
    # save encodings
    encodings = {}
    for name, param in quant_sim.model.named_parameters():
        if name.endswith("encoding_min") or name.endswith("encoding_max"):
            encodings[name] = param.data.item()

    
    def write_data(fname, datadict):
        fname = os.path.join(output_dir, fname)
        with open(fname, "w") as f:
            writer = csv.writer(f, delimiter=",")
            for name, data in datadict.items():
                _data = [name, data]
                writer.writerow(_data)
    
    write_data("encodings.csv", encodings)

    # return unwrapped model to save
    unwrapped_model = copy.deepcopy(quant_sim.model)
    modules = [module for module in unwrapped_model.modules()]
    quant_sim._remove_quantization_wrappers(unwrapped_model, modules)
    return unwrapped_model




