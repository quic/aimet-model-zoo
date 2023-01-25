#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET evaluation code for Efficientnet Lite0 '''

# general python imports
import argparse
import torch

# aimet model zoo imports
from aimet_zoo_torch.common.utils.image_net_data_loader import ImageNetDataLoader
from aimet_zoo_torch.efficientnetlite0.dataloader import eval_func, forward_pass
from aimet_zoo_torch.efficientnetlite0 import EfficientNetLite0


# add arguments
def arguments():
    parser = argparse.ArgumentParser(description='0725 changed script for efficientnet_lite0 quantization')
    parser.add_argument('--dataset-path', help='path to image evaluation dataset', type=str)
    parser.add_argument('--model-config', help='model configuration to be tested', type=str)
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--batch-size',help='batch_size for loading data',type=int,default=16)
    parser.add_argument('--use-cuda', help='Run evaluation on GPU', type=bool, default=True)
    args = parser.parse_args()
    return args


# set seed for reproducibility
def seed(seednum, use_cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seednum)
    if use_cuda:
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)

# adding hardcoded values into args from parseargs() and return config object
class ModelConfig():
    def __init__(self, args):
        self.seed=23
        self.input_shape=(1,3,224,224)
        self.checkpoint="model_efficientnetlite0_w" + str(args.default_param_bw) + "a" + str(args.default_output_bw) + "_pc_checkpoint.pth"
        self.encoding="efficientnetlite0_w" + str(args.default_param_bw) + "a" + str(args.default_output_bw) + "_pc.encodings"
        self.quant_scheme='tf_enhanced'
        self.config_file='default_config_per_channel.json'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

def main():

    args = arguments()
    config = ModelConfig(args)
    seed(seednum=23, use_cuda=args.use_cuda)

    # ===================================fp32 model ==================================
    fp32_model = EfficientNetLite0(model_config = config.model_config)
    fp32_model.from_pretrained(quantized=False)
    fp32_model.model.eval()
    fp32_acc = eval_func(fp32_model.model, config.dataset_path, config.batch_size)
    print(f'=========FP32 Model Accuracy : {fp32_acc:0.2f}% ')

    # ===================================Quantized model ==================================
    model_int8 = EfficientNetLite0(model_config = config.model_config)
    sim = model_int8.get_quantsim(quantized=True)
    encoding_dataloader = ImageNetDataLoader(config.dataset_path, image_size=224, num_samples_per_class=2)
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader.data_loader)
    quant_acc = eval_func(sim.model, config.dataset_path, config.batch_size)
    print(f'=========Quantized model Accuracy: {quant_acc:0.2f}% ')

if __name__ == '__main__':
    main()