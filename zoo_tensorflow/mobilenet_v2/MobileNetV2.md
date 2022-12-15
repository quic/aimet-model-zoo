# Tensorflow Mobilenetv2 1.4

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

**NOTE:** This model is expected **not** to work with GPUs at or after NVIDIA 30-series (e.g. RTX 3050), as those bring a new architecture not fully compatible with TF 1.X.

## Additional Dependencies

### Setup TensorFlow Models repo
- Clone the [TensorFlow Models repo](https://github.com/tensorflow/models)  
  `git clone https://github.com/tensorflow/models.git`  
  `cd models`
- Checkout this commit id:  
`git checkout 104488e40bc2e60114ec0212e4e763b08015ef97`
- Append the repo location to your `PYTHONPATH` with the following:  
`export PYTHONPATH=$PYTHONPATH:<path to tensorflow/models repo>/research/slim`

## Model checkpoint and dataset
- Downloading model checkpoint and config file are handled by evaluation script.
- The optimized Mobilenet v2 1.4 checkpoint can be downloaded from [Releases](/../../releases).
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config.json](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset
- ImageNet can be downloaded from here:
  - http://www.image-net.org/
- For this evaluation, Tf-records of ImageNet validation dataset are required. (See https://github.com/tensorflow/models/tree/master/research/slim#Data for details)
- The Tf-records of ImageNet validation dataset should be organized in the following way
```bash
< path to ImageNet validation dataset Tf-records>
├── validation-00000-of-00128
├── validation-00001-of-00128
├── ...
```

## Usage
- To run evaluation with QuantSim in AIMET, use the following:
```bash
python mobilenet_v2_140_quanteval.py \
    --dataset-path <path to imagenet validation TFRecords> \
    --batch-size <batch size for loading the dataset> \
    --model-to-eval <which model to evaluate. Two options are available: 'fp32' for evaluating original fp32 model, 'int8' for evaluating quantized int8 model.>
```

## Quantization configuration 
In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Operations which shuffle data such as reshape or transpose do not require additional quantizers
