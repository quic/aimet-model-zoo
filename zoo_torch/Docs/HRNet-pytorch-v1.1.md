# HRNet-pytorch-v1.1

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further. This model was tested using AIMET version 1.21.0.

## Experiment setup
Follow [HRNet pytorch-v1.1 branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/README.md#quick-start) installation instructions.

## Obtain model checkpoints, dataset and configuration
- The original HRNet checkpoint can be downloaded from here:
  - https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1
- Optimized HRNet checkpoint can be downloaded from the [Releases](/../../releases) page.
- Cityscapes can be downloaded from here:
  - https://www.cityscapes-dataset.com/
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.qualcomm.com/qualcomm-ai/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, copy eval_hrnet.py into HRNet tools directory and use the following:
```bash
python tools/eval_hrnet.py \
	--quant-scheme <Quant scheme to use for quantization> \
	--output-bw <Default output bitwidth for quantization> \
	--default-param-bw <Default parameter bitwidth for quantization> \
	--config-file <Quantsim configuration file> \
	--checkpoint-prefix <optimized checkpoint and encodings prefix> \
	--cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml TEST.FLIP_TEST False
```

## Quantization Configuration
- Weight quantization: 8 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used for weight quantization scheme
- TF_enhanced was used for activation quantization scheme
- Cross layer equalization and Adaround have been applied on optimized checkpoint
