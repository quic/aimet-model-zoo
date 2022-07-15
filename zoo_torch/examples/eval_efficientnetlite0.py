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

#torch imports
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import geffnet

#aimet imports
from aimet_torch.quantsim import load_checkpoint
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

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

    #Define transformation
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
def eval_func(model, DATA_DIR, BATCH_SIZE=64):

    #Get Dataloader
    dataloader_eval = get_imagenet_dataloader(DATA_DIR,BATCH_SIZE)

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

    return float(100* correct / total_samples)

# Forward pass for encoding calculations
def forward_pass(model, DATA_DIR):
    #Get Dataloader
    dataloader_encoding = get_imagenet_dataloader(DATA_DIR)

    on_cuda = next(model.parameters()).is_cuda

    with torch.no_grad():
        for data, _ in dataloader_encoding:
            if on_cuda:
                data= data.cuda()

            output = model(data)

# add arguments
def arguments():
    parser = argparse.ArgumentParser(description='script for efficientnet_lite0 quantization')
    parser.add_argument('--checkpoint', help='Path to optimized checkpoint', default=None, type=str)
    parser.add_argument('--encodings', help='Path to optimized encodings', default=None, type=str)
    parser.add_argument('--use_cuda', help='Use cuda', default=True, type=bool)
    parser.add_argument('--calibration_dataset', help='path to calibration dataset',type=str)
    parser.add_argument('--evaluation_dataset', help='path to evaluation dataset',type=str)
    parser.add_argument('--seed', help='Seed number for reproducibility', default=1000)
    parser.add_argument('--input-shape', help='Model input shape for quantization.', type = tuple, default=(1,3,224,224))
    parser.add_argument('--quant-scheme', help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--config-file', help='Quantsim configuration file.', default=None, type=str)
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


def main():
    #Load parameters from arguments
    args = arguments()
    seed(args.seed, args.use_cuda)

    # Get fp32 model and convert to eval mode
    model = getattr(geffnet, 'efficientnet_lite0')(pretrained=True)
    model.eval()

    if args.use_cuda:
        model.cuda()

    #Print FP32 accuracy
    fp32_acc = eval_func(model, args.evaluation_dataset)
    print(f'FP32 accuracy: {fp32_acc:0.2f}%')

    # Get optimized fp32 model by loading checkpoint
    if args.checkpoint:
        print(args.checkpoint)
        ada_model = torch.load(args.checkpoint)
    else:
        raise ValueError('checkpoint path {} must be specified'.format(args.checkpoint_path))
    ada_model.eval()

    if args.use_cuda:
        ada_model.cuda()

    # Initial Quantized model
    if hasattr(args, 'quant_scheme'):
        if args.quant_scheme == 'range_learning_tf':
            quant_scheme = QuantScheme.training_range_learning_with_tf_init
        elif args.quant_scheme == 'range_learning_tfe':
            quant_scheme = QuantScheme.training_range_learning_with_tf_enhanced_init
        elif args.quant_scheme == 'tf':
            quant_scheme = QuantScheme.post_training_tf
        elif args.quant_scheme == 'tf_enhanced':
            quant_scheme = QuantScheme.post_training_tf_enhanced
        else:
            raise ValueError("Got unrecognized quant_scheme: " + args.quant_scheme)
        if args.use_cuda:
            dummy_input = torch.rand(args.input_shape, device = 'cuda')
        else:
            dummy_input = torch.rand(args.input_shape)
        kwargs = {
            'quant_scheme': quant_scheme,
            'default_param_bw': args.default_param_bw,
            'default_output_bw': args.default_output_bw,
            'config_file': args.config_file,
            'dummy_input': dummy_input
        }
    sim = QuantizationSimModel(ada_model, **kwargs)

    # Get optimized encodings
    if args.encodings:
        print(args.encodings)
        sim.set_and_freeze_param_encodings(encoding_path=args.encodings)
    else:
        raise ValueError('encodings path {} must be specified'.format(args.encodings))

    sim.compute_encodings(forward_pass, forward_pass_callback_args=args.calibration_dataset)

    quant_acc = eval_func(sim.model, args.evaluation_dataset)
    print(f'Quantized W8A8 accuracy: {quant_acc:0.2f}%')


if __name__ == '__main__':
    main()



