# PyTorch MobileNetV2-SSD-lite

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Model modifications
1. Clone the original repository
```bash
git clone https://github.com/qfgaohao/pytorch-ssd.git
cd pytorch-ssd
git checkout f61ab424d09bf3d4bb3925693579ac0a92541b0d
git apply ../aimet-model-zoo/zoo_torch/examples/ssd_mobilenetv2/patch_ssd.patch
```
2. Add AIMET model zoo and Pytorch-SSD to the python path
```
export PYTHONPATH=$PYTHONPATH:<path to parent>/pytorch-ssd
export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo
```

## Obtaining model checkpoint and dataset
- The original MobileNetV2-SSD-lite checkpoint can be downloaded from here:
  - https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
- Optimized checkpoint can be downloaded from the [Releases](/../../releases).
- Pascal VOC2007 dataset can be downloaded from here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html


## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python ssd_mobilenetv2_quanteval.py --dataset-path <The root directory of dataset, e.g., my_path/VOCdevkit/VOC2007/>
```

## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used as quantization scheme
- Cross-layer-Equalization and Adaround have been applied on optimized checkpoint
