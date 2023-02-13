# Tensorflow EfficientNet Lite-0

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

## Additional Dependencies
### Setup TensorFlow TPU repo
- Clone the [TensorFlow TPU repo](https://github.com/tensorflow/tpu)  
```bash
git clone https://github.com/tensorflow/tpu.git
cd tpu
git checkout c75705856290a4119d609110956442449d73e0a5
```
- Append the repo location to your `PYTHONPATH` with the following:  
```bash
export PYTHONPATH=$PYTHONPATH:<path to TPU repo>/tpu/models/official/efficientnet
```

## Model checkpoint and dataset
- Downloading checkpoints and Quantization configuration file are handled through evaluation script.
- The original EfficientNet Lite-0 checkpoint can be downloaded from here:
  -- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
- Optimized EfficientNet Lite-0 checkpoint can be downloaded from [Releases](/../../releases).
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config.json](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset 
- ImageNet can be downloaded from here:
  - https://image-net.org/download
- 2012 ImageNet validation dataset is used for this evaluation script. The data should be organized in the following way
- Imagenet validation label file ILSVRC2012_validation_ground_truth.txt is required for evaluation. This file should only contain labels of images. 
- ILSVRC2012_validation_ground_truth.txt existing in ImageNet validation dataset has two columns. 1st column is ImageID, the 2nd column is labels of corresponding images. To get only the 2nd column, use ` awk '{ print $2}' ILSVRC2012_validation_ground_truth.txt > ILSVRC2012_validation_ground_truth.only_labels.txt `

```bash 
<imagenet validation dataset path>/
├── ILSVRC2012_val_00000001.JPEG
├── ILSVRC2012_val_00000002.JPEG
├── ...
```

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python efficientnet_quanteval.py \
    --imagenet-eval-glob=<imagenet eval glob> \
    --imagenet-eval-label=<imagenet validation labels file> \
    --model-to-eval < which model to evaluate. Two options are available: 'fp32' for evaluating original fp32 model, 'int8' for evaluating quantized int8 model.>
```

## Quantization configuration 

In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Operations which shuffle data such as reshape or transpose do not require additional quantizers
