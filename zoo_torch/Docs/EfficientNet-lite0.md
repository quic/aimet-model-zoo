# PyTorch-EfficientNet-lite0

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.21/packaging/install.md) before proceeding further. This model was tested using AIMET version 1.21.0.

## Additional Dependencies
1. Install geffnet using pip install
```
sudo -H pip install geffnet
```

## Obtain model checkpoints, dataset and configuration
- The original EfficientNet-lite0 checkpoint can be downloaded from here:
  - https://github.com/rwightman/gen-efficientnet-pytorch
- Optimized EfficientNet-lite0 checkpoint can be downloaded from the [Releases](/../../releases) page.
- ImageNet can be downloaded from here:
  - http://www.image-net.org/
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.qualcomm.com/qualcomm-ai/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_efficientnetlite0.py \
		--checkpoint <path to optimiezd checkpoint> \
		--encodings <Path to optimized encodings> \
		--use_cuda <Use cuda or cpu> \
		--calibration_dataset <path to calibration dataset> \
		--evaluation_dataset <path to evaluation dataset> \
		--seed <Seed number for reproducibility> \
		--input-shape <Model input shape for quantization>
		--quant-scheme <Quant scheme to use for quantization>
		--default-output-bw <Default output bitwidth for quantization> \
		--default-param-bw <Default parameter bitwidth for quantization> \
		--config-file <Quantsim configuration file>
```

## Quantization Configuration
- Weight quantization: 8 bits per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used for weight quantization scheme
- TF was used for activation quantization scheme
- Batch norm folding and Adaround have been applied on optimized efficientnet-lite checkpoint
- [Conv - Relu6] layers has been fused as one operation via manual configurations
- 4K Images (4 images per class) from ImageNet traing dataset are used as calibration dataset
- Standard ImageNet validation dataset are usef as evaluation dataset
