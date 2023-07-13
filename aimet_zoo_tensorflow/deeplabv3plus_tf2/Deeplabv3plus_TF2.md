# Tensorflow DeeplabV3plus for TensorFlow 2.4

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.25 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.25.0) i.e. please set `release_tag="1.25"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

## Additional Dependencies
pip install matplotlib==3.2.1

## Model checkpoint and dataset
The TF2 pretrained DeeplabV3plus can be downloaded at release page

## Dataset
The Pascal Dataset can be downloaded from here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

## Usage
- To run evaluation with QuantSim in AIMET, use the following:
```bash
python aimet_zoo_tensorflow/deeplabv3plus_tf2/evaluators/deeplabv3plus_tf2_quanteval.py \
    --dataset-path <path to pascal dataset> \
    --batch-size <batch size for loading the dataset> \
    --model-config <model configuration to test>
```
Available model configurations are:

- deeplabv3plus_xception_w8a8

- deeplabv3plus_mbnv2_w8a8

- Example : python aimet_zoo_tensorflow/deeplabv3plus_tf2/evaluators/deeplabv3plus_tf2_quanteval.py --dataset-path <imagenet dataset path> --batch-size 4 --model-config deeplabv3plus_xception_w8a8

## Quantization configuration 
In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- For MobileNetV2 backbone, Quantization Aware Traning has been performed on the optimized checkpoint

## Results
Below are the *mIoU* results of the TensorFlow 2 DeeplabV3plus_xception model for the VOC2012 dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>mIoU (%)</th>
  </tr>
  <tr>
    <td>DeeplabV3plus_xception_FP32</td>
    <td>87.71</td>
  </tr>
  <tr>
    <td>DeeplabV3plus_xception + simple PTQ(w8a8)</td>
    <td>87.21</td>
  </tr>
</table>

Below are the *mIoU* results of the TensorFlow 2 DeeplabV3plus_mbnv2 model for the VOC2012 dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>mIoU (%)</th>
  </tr>
  <tr>
    <td>DeeplabV3plus_mbnv2_FP32</td>
    <td>72.28</td>
  </tr>
  <tr>
    <td>DeeplabV3plus_mbnv2 + QAT(w8a8)</td>
    <td>71.71</td>
  </tr>
</table>

