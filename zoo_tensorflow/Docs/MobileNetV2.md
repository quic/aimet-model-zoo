# Mobilenetv2 1.4

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies

### Setup TensorFlow Models repo
- Clone the [TensorFlow Models repo](https://github.com/tensorflow/models)  
  `git clone https://github.com/tensorflow/models.git`

- checkout this commit id:  
  `git checkout 104488e40bc2e60114ec0212e4e763b08015ef97`

- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=$PYTHONPATH:<path to tensorflow/models repo>/research/slim`

## Obtaining model checkpoint and dataset
- The optimized Mobilenet v2 1.4 checkpoint can be downloaded from [Releases](/../../releases).
- ImageNet can be downloaded here:
  - http://www.image-net.org/

## Usage
- To run evaluation with QuantSim in AIMET, use the following:
```bash
python mobilenet_v2_140_quanteval.py \
    --model-name=mobilenet_v2_140 \
    --checkpoint-path=<path to mobilenet_v2_140 checkpoint> \
    --dataset-dir=<path to imagenet validation TFRecords> \
    --quantsim-config-file=<path to config file with symmetric weights>
```

- If you are using a model checkpoint which has Batch Norms already folded (such as the optimized model checkpoint), please specify the `--ckpt-bn-folded` flag:

```bash
python mobilenet_v2_140_quanteval.py \
    --model-name=mobilenet_v2_140 \
    --checkpoint-path=<path to mobilenet_v2_140 checkpoint> \
    --dataset-dir=<path to imagenet validation TFRecords> \
    --quantsim-config-file=<path to config file with symmetric weights>
    --ckpt-bn-folded
```

## Quantizer Op Assumptions
In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- Operations which shuffle data such as reshape or transpose do not require additional quantizers