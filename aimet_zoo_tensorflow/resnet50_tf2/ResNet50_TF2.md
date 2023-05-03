# Tensorflow ResNet50 for TensorFlow 2.4

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.25 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.25) i.e. please set `release_tag="1.25"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

## Additional Dependencies
pip install numpy==1.19.5

## Model checkpoint and dataset
The TF2 pretrained resnet50 model is directly imported from package tensorflow.keras.applications

## Dataset
- ImageNet can be downloaded from here:
  - http://www.image-net.org/
- The directory where the data is located should contains subdirectories, each containing images for a class
- The ImageNet validation dataset should be organized in the following way
```bash
< path to ImageNet validation dataset >
├── n01440764
├── n01443537
├── ...
```

## Usage
- To run evaluation with QuantSim in AIMET, use the following:
```bash
python aimet_zoo_tensorflow/resnet50_tf2/evaluators/resnet50_tf2_quanteval.py \
    --dataset-path <path to imagenet dataset> \
    --batch-size <batch size for loading the dataset> \
    --model-config <model configuration to test>
```
Available model configurations are:

- resnet50_w8a8

- Example : python aimet_zoo_tensorflow/resnet50_tf2/evaluators/resnet50_tf2_quanteval.py --dataset-path <imagenet dataset path> --batch-size 4 --model-config resnet50_w8a8

## Quantization configuration 
In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized

## Results
Below are the *top1 accuracy* results of the TensorFlow 2.4 resnet50 model for the imagenet dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>Top1 (%)</th>
  </tr>
  <tr>
    <td>Resnet50_FP32</td>
    <td>74.9</td>
  </tr>
  <tr>
    <td>Resnet50 + simple PTQ(w8a8)</td>
    <td>74.8</td>
  </tr>
</table>
