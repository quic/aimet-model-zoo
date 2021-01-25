# PyTorch-MobileNetV2-SSD-lite

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Model modifications
1. Clone the original repository
```
git clone https://github.com/qfgaohao/pytorch-ssd.git
cd pytorch-ssd
git checkout f61ab424d09bf3d4bb3925693579ac0a92541b0d
git apply ../aimet-model-zoo/zoo_torch/examples/torch_ssd_eval.patch
```
2. Place the model definition & eval_ssd.py to aimet-model-zoo/zoo_torch/examples/
```
mv vision ../aimet-model-zoo/zoo_torch/examples/
mv eval_ssd.py ../aimet-model-zoo/zoo_torch/examples/
```


## Obtaining model checkpoint and dataset
- The original MobileNetV2-SSD-lite checkpoint can be downloaded here:
  - https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
- Optimized checkpoint can be downloaded from the [Releases](/../../releases).
- Pascal VOC2007 dataset can be downloaded here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_ssd.py \
 --net <Architecture to run, currently only 'mb2-ssd-lite' is supported> \
 --trained_model <Path to checkpoint to load> \
 --dataset <The root directory of dataset> \
 --label_file <Path to label file to parse> \
 --eval_dir <Path to save a result>
```

## Quantization Configuration
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Cross-layer-Equalization and Adaround have been applied on optimized checkpoint
