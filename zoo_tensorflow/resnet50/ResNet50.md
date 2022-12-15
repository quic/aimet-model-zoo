# TensorFlow ResNet 50

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

## Environment Setup
This model requires the following python package versions:  
```
pip install tensorflow-gpu==1.15.0  
```

- Clone the [TensorFlow Models repo](https://github.com/tensorflow/models)  
  `git clone https://github.com/tensorflow/models.git`  
  `cd models`

- Checkout this commit id:

  `git checkout 104488e40bc2e60114ec0212e4e763b08015ef97`

- Append the repo location to your `PYTHONPATH` with the following:

  `export PYTHONPATH=$PYTHONPATH:<path to tensorflow models repo>/research/slim`  
  `export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo`

**Note:** This model is expected **not** to work with GPUs at or after NVIDIA 30-series (e.g. RTX 3050), as those bring a new architecture not fully compatible with TF 1.X

## Dataset
- ImageNet can be downloaded from here:
  - http://www.image-net.org/

The dataset must then be converted to TFRecords formatting. The resulting directory should contain TFRecords files named as: `train-00000-of-01024` up to `train-01023-of-01024` and `validation-00001-of-00128` up to `validation-00127-of-00128`. This evalution script will only use validation data.

## Model Weights
- The original ResNet 50 checkpoint is available on the [TensorFlow Models repo](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz).

## Usage
```bash
python resnet50_v1_quanteval.py  \
    --dataset-path <path to imagenet validation TFRecords>  \
    --eval_quantized <True evaluates the optimized model, False the original model>
```
Setting `eval_quantized=True` will evaluate the optimized model's performance on both GPU and on a simulated hardware device.
Similarly, `eval_quantized=False` will evalute the original source model on both GPU and simulated device.

## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are quantized
- Activation quantization: 8 bits, asymmetric quantization
- Operations which shuffle data such as reshape or transpose do not require additional quantizers
