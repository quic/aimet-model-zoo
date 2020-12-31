# ResNet 50

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

- The original ResNet 50 checkpoint can be downloaded from [TensorFlow Models repo](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz).

- ImageNet can be downloaded here:
  - http://www.image-net.org/



## Usage

- To run evaluation with QuantSim in AIMET, use the following

```bash
python resnet_v1_50_quanteval.py \
    --model-name=resnet_v1_50 \
    --checkpoint-path=<path to resnet_v1_50 checkpoint> \
    --dataset-dir=<path to imagenet validation TFRecords> \
    --quantsim-config-file=<path to config file with symmetric weights>
```

- If you are using a model checkpoint which has Batch Norms already folded, please specify the `--ckpt-bn-folded` flag:

```bash
python resnet_v1_50_quanteval.py \
    --model-name=resnet_v1_50 \
    --checkpoint-path=<path to resnet_v1_50 checkpoint> \
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
