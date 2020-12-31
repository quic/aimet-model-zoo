# PyTorch-EfficientNet-lite0

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies
1. Install geffnet using pip install
```
sudo -H pip install geffnet
```
## Obtaining model checkpoint and dataset

- The original EfficientNet-lite0 checkpoint can be downloaded from here:
  - https://github.com/rwightman/gen-efficientnet-pytorch
- Optimized EfficientNet-lite0 checkpoint can be downloaded from the [Releases](/../../releases) page.
- ImageNet can be downloaded from here:
  - http://www.image-net.org/

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_efficientnetlite0.py \
		--checkpoint <path to optimiezd checkpoint> \
		--images-dir <path to imagenet root directory> \
		--quant-scheme <quantization schme to run>  \
		--quant-tricks <preprocessing steps prior to Quantization> \
		--default-output-bw <bitwidth for activation quantization> \
		--default-param-bw <bitwidth for weight quantization>
```

## Quantization Configuration
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Batch norm folding and Adaround has been applied on optimized efficientnet-lite checkpoint
- [Conv - Relu6] layers has been fused as one operation via manual configurations
