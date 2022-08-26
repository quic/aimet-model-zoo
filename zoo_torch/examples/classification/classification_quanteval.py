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
import urllib.request
import wget

#Torch related imports
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models

#AIMET torch related imports
from aimet_torch.quantsim import QuantizationSimModel
from classification_utils.image_net_data_loader import ImageNetDataLoader



def get_imagenet_dataloader(image_dir, BATCH_SIZE=64):
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


# add arguments
def arguments():
    parser = argparse.ArgumentParser(description='script for classification model quantization')
    parser.add_argument('--fp32-model', help='FP32 torchvision model to load', default=None, choices = ['resnet18', 'resnet50', 'regnet_x_3_2gf'], type=str, required=True)
    parser.add_argument('--default-param-bw', help='weight bitwidth for quantization', default=8, type=int)
    parser.add_argument('--default-output-bw', help='output bitwidth for quantization', default=8, type=int)
    parser.add_argument('--use-cuda', help='Use cuda', default=True, type=bool)
    parser.add_argument('--evaluation-dataset', help='path to evaluation dataset',type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    #Load parameters from arguments
    args = arguments()
    
    # Get fp32 model and convert to eval mode
    model = getattr(torchvision.models,args.fp32_model)(pretrained=True)
    model.eval()

    
    #updata use-cuda args based on availability of cuda devices
    use_cuda = args.use_cuda and torch.cuda.is_available()
    
    #Define prefix
    prefix = f'{args.fp32_model}_W{args.default_param_bw}A{args.default_output_bw}'

    # Download weights for optimized model and load optimized model and encodings
    print('Downloading optimized model weights')
    URL = f"https://github.com/quic/aimet-model-zoo/releases/download/torchvision_classification_INT4%2F8/"
    wget.download(URL+f'{prefix}.pth', f'./{prefix}.pth')
    wget.download(URL+f'{prefix}.encodings', f'./{prefix}.encodings')

    #Download aimet config file
    URL = 'https://raw.githubusercontent.com/quic/aimet/develop/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json'
    wget.download(URL,'./default.json')
    
    args.aimet_config_file = './default.json'
    
    optimized_model = torch.load(f"./{prefix}.pth")
    optimized_encodings_path = f"./{prefix}.encodings"
    
    if use_cuda:
        model.cuda()

    #Print FP32 accuracy
    fp32_acc = eval_func(model, args.evaluation_dataset)
    print(f'FP32 accuracy: {fp32_acc:0.3f}%')

    #create quantsim from checkpoint
    #Define dummy input for quantsim
    dummy_input = torch.randn((1, 3, 224, 224))

    #Move Optimized model to eval mode
    optimized_model.eval()
    
    if use_cuda:
        optimized_model.cuda()
        dummy_input = dummy_input.cuda()

    #Create quantsim using appropriate weight bitwidth for quantization
    sim = QuantizationSimModel(optimized_model, quant_scheme='tf_enhanced',default_param_bw=args.default_param_bw,default_output_bw=args.default_output_bw, dummy_input=dummy_input, config_file=args.aimet_config_file)

    #Set and freeze optimized weight encodings
    sim.set_and_freeze_param_encodings(encoding_path=optimized_encodings_path)
    
    
    #Define AIMET torch imagenet dataloader to pick 2 images from each class
    encoding_dataloader = ImageNetDataLoader(args.evaluation_dataset,image_size=224,num_samples_per_class=2)

    #Compute activation encodings
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader.data_loader)

    quant_acc = eval_func(sim.model.cuda(), args.evaluation_dataset)
    print(f'Quantized quantized accuracy: {quant_acc:0.3f}%')

if __name__ == '__main__':
    main()



