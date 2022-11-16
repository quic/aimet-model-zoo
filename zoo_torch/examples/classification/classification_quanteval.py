#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' AIMET Quantsim evaluation code for quantized classification models - Resnet18, Resnet50, Regnet_x_3_2gf '''

#General Imports
import argparse
import random
import sys, os, tarfile
import urllib.request

#Torch related imports
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models


#AIMET torch related imports
from aimet_torch.quantsim import QuantizationSimModel
from zoo_torch.examples.common.image_net_data_loader import ImageNetDataLoader
from aimet_model_zoo.zoo_torch.common.utils import get_device


QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22.1/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json"
OPTIMIZED_CHECKPOINT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torchvision_classification_INT4%2F8/"

def download_weights(prefix):
    # Download config file
    if not os.path.exists("./default_config_per_channel.json"):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")

    # Download optimized model
    if not os.path.exists(f"./{prefix}.pth"):
        urllib.request.urlretrieve(f"{OPTIMIZED_CHECKPOINT_URL}/{prefix}.pth", f"{prefix}.pth")
    if not os.path.exists(f"./{prefix}.encodings"):
        urllib.request.urlretrieve(f"{OPTIMIZED_CHECKPOINT_URL}/{prefix}.encodings",f"{prefix}.encodings")


def get_imagenet_dataloader(image_dir, BATCH_SIZE=128):
    '''
    Helper function to get imagenet dataloader from dataset directory
    '''
    def generate_dataloader(data, name, transform, batch_size=32):
        if data is None:
            return None

        if transform is None:
            dataset = torchvision.datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = torchvision.datasets.ImageFolder(data, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

        return dataloader

    #Define transformation
    preprocess_transform_pretrain = T.Compose([
        T.Resize(256),  # Resize images to 256 x 256
        T.CenterCrop(224),  # Center crop image
        T.ToTensor(),  # Converting cropped images to tensors
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataloader = generate_dataloader(image_dir, "train",transform=preprocess_transform_pretrain, batch_size=BATCH_SIZE)
    return dataloader

def eval_func(model, DATA_DIR,BATCH_SIZE=128):
    '''
    Evaluates the model on validation dataset and returns the classification accuracy
    '''

    #Get Dataloader
    dataloader_eval = get_imagenet_dataloader(DATA_DIR,BATCH_SIZE)
    
    model.eval()

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
    
    del dataloader_eval

    return float(100* correct / total_samples)

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

def seed(seed_num, args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if args.use_cuda:
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

# add arguments
def arguments():
    parser = argparse.ArgumentParser(description='script for classification model quantization')
    parser.add_argument('--fp32-model', help='FP32 torchvision model to load', default=None, choices = ['resnet18', 'resnet50', 'regnet_x_3_2gf'], type=str, required=True)
    parser.add_argument('--default-param-bw', help='weight bitwidth for quantization', default=8, type=int)
    parser.add_argument('--default-output-bw', help='output bitwidth for quantization', default=8, type=int)
    parser.add_argument('--use-cuda', help='Use cuda', default=True, type=bool)
    parser.add_argument('--evaluation-dataset', help='path to evaluation dataset',type=str, required=True)
    parser.add_argument('--batch-size', help='Data batch size for a model', type = int, default=128)
    parser.add_argument('--config-file', help='Data batch size for a model', type = str, default='./default_config_per_channel.json')
    args = parser.parse_args()
    return args


def main():
    #Load parameters from arguments
    args = arguments()

    # Set seed value
    seed(0, args)

    # Get fp32 model and convert to eval mode
    model = getattr(torchvision.models,args.fp32_model)(pretrained=True)
    model.eval()

    
    #get device
    device = get_device(args)

    #Define prefix
    prefix = f'{args.fp32_model}_W{args.default_param_bw}A{args.default_output_bw}'

    # Download weights for optimized model and load optimized model and encodings
    download_weights(prefix)

    optimized_model = torch.load(f"./{prefix}.pth")
    optimized_encodings_path = f"./{prefix}.encodings"

    model.to(device)

    #Print FP32 accuracy
    fp32_acc = eval_func(model, args.evaluation_dataset, args.batch_size)

    #create quantsim from checkpoint
    #Define dummy input for quantsim
    dummy_input = torch.randn((1, 3, 224, 224))

    #Move Optimized model to eval mode
    optimized_model.eval()
    

    optimized_model.to(device)
    dummy_input = dummy_input.to(device)

    #Create quantsim using appropriate weight bitwidth for quantization
    sim = QuantizationSimModel(optimized_model, quant_scheme='tf_enhanced',default_param_bw=args.default_param_bw,default_output_bw=args.default_output_bw, dummy_input=dummy_input, config_file=args.config_file)

    #Set and freeze optimized weight encodings
    sim.set_and_freeze_param_encodings(encoding_path=optimized_encodings_path)
    
    
    #Define AIMET torch imagenet dataloader to pick 2 images from each class
    encoding_dataloader = ImageNetDataLoader(args.evaluation_dataset,image_size=224,num_samples_per_class=2)

    #Compute activation encodings
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader.data_loader)

    quant_acc = eval_func(sim.model.cuda(), args.evaluation_dataset)

    #Print accuracy stats
    print("Evaluation Summary:")
    print(f"Original Model | Accuracy on 32-bit device: {fp32_acc:.4f}")
    print(f"Optimized Model | Accuracy on {args.default_param_bw}-bit device: {quant_acc:.4f}")


if __name__ == '__main__':
    main()



