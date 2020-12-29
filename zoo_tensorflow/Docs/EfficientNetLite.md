# EfficientNet Lite-0

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies
### Setup TensorFlow TPU repo
- Clone the [TensorFlow TPU repo](https://github.com/tensorflow/tpu)  
  `git clone https://github.com/tensorflow/tpu.git`
- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=$PYTHONPATH:<path to TPU repo>/tpu/models/official/efficientnet`

## Obtaining model checkpoint and dataset
- The original EfficientNet Lite-0 checkpoint can be downloaded here:
  - https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
- Optimized EfficientNet Lite-0 checkpoint can be downloaded from [Releases](/../../releases).
- ImageNet can be downloaded here:
  - http://www.image-net.org/


## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python efficientnet_quanteval.py
    --model-name=efficientnet-lite0
    --checkpoint-path=<path to efficientnet-lite0 checkpoint>
    --imagenet-eval-glob=<imagenet eval glob>
    --imagenet-eval-label=<imagenet validation labels file>
    --quantsim-config-file=<path to config file with symmetric weights>
```

- If you are using a model checkpoint which has Batch Norms already folded (such as the optimized model checkpoint), please specify the `--ckpt-bn-folded` flag:
```bash
python efficientnet_quanteval.py
    --model-name=efficientnet-lite0
    --checkpoint-path=<path to efficientnet-lite0 checkpoint>
    --imagenet-eval-glob=<imagenet eval glob>
    --imagenet-eval-label=<imagenet validation labels file>
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