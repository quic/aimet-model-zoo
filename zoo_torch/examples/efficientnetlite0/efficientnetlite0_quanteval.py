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
import os
import argparse
import urllib.request

# torch imports
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import geffnet
import time

# aimet imports
from aimet_torch.quantsim import load_checkpoint
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

# aimet model zoo imports
from zoo_torch.examples.common import utils
from zoo_torch.examples.common.image_net_data_loader import ImageNetDataLoader

# ImageNet data loader
def get_imagenet_dataloader(image_dir, BATCH_SIZE=64):
    def generate_dataloader(data, transform, batch_size=BATCH_SIZE):
        if data is None:
            return None

        if transform is None:
            dataset = torchvision.datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = torchvision.datasets.ImageFolder(data, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

        return dataloader

    # Define transformation
    preprocess_transform_pretrain = T.Compose([
        T.Resize(256),  # Resize images to 256 x 256
        T.CenterCrop(224),  # Center crop image
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Converting cropped images to tensors
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataloader = generate_dataloader(image_dir, transform=preprocess_transform_pretrain, batch_size=BATCH_SIZE)
    return dataloader


# Evaluates the model on validation dataset and returns the classification accuracy
def eval_func(model, DATA_DIR, BATCH_SIZE=16):
    # Get Dataloader
    dataloader_eval = get_imagenet_dataloader(DATA_DIR, BATCH_SIZE)

    correct = 0
    total_samples = 0
    on_cuda = next(model.parameters()).is_cuda

    with torch.no_grad():
        for data, label in dataloader_eval:
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)
            _, prediction = torch.max(output, 1)
            correct += (prediction == label).sum()
            total_samples += len(output)

    return float(100 * correct / total_samples)


# Forward pass for encoding calculations
def pass_calibration_data(model, args):
    # Get Dataloader

    dataloader_encoding = get_imagenet_dataloader(args['evaluation_dataset'])
    on_cuda = next(model.parameters()).is_cuda
    model.eval()
    batch_counter = 0
    samples = 100  # number of samples for validation
    with torch.no_grad():
        for input_data, target_data in dataloader_encoding:
            if on_cuda:
                input_data = input_data.cuda()

            output_data = model(input_data)
            batch_counter += 1
            if (batch_counter * args['batch_size']) > samples:
                break

def forward_pass(model, dataloader):
    '''
    forward pass through the calibration dataset
    '''
    model.eval()

    on_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for data, label in dataloader:
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)


    del dataloader



# add arguments
def arguments():
    parser = argparse.ArgumentParser(description='0725 changed script for efficientnet_lite0 quantization')
    parser.add_argument('--dataset-path', help='path to image evaluation dataset', type=str)
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

def download_weights():
    if not os.path.exists("./default_config_per_channel.json"):
        url_checkpoint = 'https://raw.githubusercontent.com/quic/aimet/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json'
        urllib.request.urlretrieve(url_checkpoint, "default_config_per_channel.json")
    if not os.path.exists("./efficientnetlite0_w8a8_pc.encodings"):
        url_encoding = "https://github.com/quic/aimet-model-zoo/releases/download/torch_effnet_lite0_w8a8_pc/efficientnetlite0_w8a8_pc.encodings"
        urllib.request.urlretrieve(url_encoding, "efficientnetlite0_w8a8_pc.encodings")
    if not os.path.exists("model_efficientnetlite0_w8a8_pc_checkpoint.pth"):
        url_config = "https://github.com/quic/aimet-model-zoo/releases/download/torch_effnet_lite0_w8a8_pc/model_efficientnetlite0_w8a8_pc_checkpoint.pth"
        urllib.request.urlretrieve(url_config, "model_efficientnetlite0_w8a8_pc_checkpoint.pth")

# adding hardcoded values into args from parseargs() and return config object
class ModelConfig():
    def __init__(self, args):
        self.seed=23
        self.input_shape=(1,3,224,224)
        self.checkpoint='model_efficientnetlite0_w8a8_pc_checkpoint.pth'
        self.encoding='efficientnetlite0_w8a8_pc.encodings'
        self.quant_scheme='tf_enhanced'
        self.config_file='default_config_per_channel.json'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

def main():
    # Load parameters from arguments
    args = arguments()

    # Adding hardcoded values to config on top of args
    config=ModelConfig(args)

    download_weights()
        
    device=utils.get_device(args)

    seed(config.seed, config.use_cuda)

    # Initial Quantized model


    dummy_input = torch.rand(config.input_shape, device=device)

    kwargs = {
        'quant_scheme': QuantScheme.post_training_tf_enhanced,
        'default_param_bw': config.default_param_bw,
        'default_output_bw': config.default_output_bw,
        'config_file': config.config_file,
        'dummy_input': dummy_input
    }
    kwargs_encoding = {'evaluation_dataset': config.dataset_path,
                       'batch_size': config.batch_size}

    # ===================================fp32 model ==================================
    # Get fp32 model and convert to eval mode

    fp32_model = getattr(geffnet, 'efficientnet_lite0')(pretrained=True)
    fp32_model.eval()
    fp32_model.to(device)

    # Print FP32 accuracy
    fp32_acc = eval_func(fp32_model, config.dataset_path, config.batch_size)
    print(f'=========FP32 Model Accuracy : {fp32_acc:0.2f}% ')

    # ===================================Quantized model ==================================
    # Get quantized model and convert to eval mode
    # Get quantized model by loading checkpoint
    try:
        model_int8 = torch.load(config.checkpoint)
    except:
        raise ValueError("checkpoint file does not exist, check download_weights")
    model_int8.eval()
    model_int8.to(device)
    # Create quantsim using appropriate weight bitwidth for quantization
    sim = QuantizationSimModel(model_int8, **kwargs)
    # Get optimized encodings
    # Set and freeze encodings to use same quantization grid and then invoke compute encoding
    sim.set_and_freeze_param_encodings(encoding_path=config.encoding)

    #Define AIMET torch imagenet dataloader to pick 2 images from each class
    encoding_dataloader = ImageNetDataLoader(config.dataset_path,image_size=224,num_samples_per_class=2)

    #Compute activation encodings
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader.data_loader)

    quant_acc = eval_func(sim.model, config.dataset_path, config.batch_size)
    print(f'=========Quantized W8A8 model Accuracy: {quant_acc:0.2f}% ')

if __name__ == '__main__':
    main()

