# PyTorch MobileNetV2-SSD-lite

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.23.

### Install dependencies 
```bash 
   python -m pip install pycocotools
```
Append the repo location to your `PYTHONPATH` with the following:  
  ```bash
  export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo>
  ```

### Dataset
Pascal VOC2007 dataset can be downloaded from here:
- http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python3  aimet_zoo_torch/ssd_mobilenetv2/evaluators/ssd_mobilenetv2_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to the downloaded Pascal dataset, should end in VOCdevkit/VOC2007> 
```

Available model configurations are:
- ssd_mobilenetv2_w8a8

---

## Obtaining model checkpoint
- The original MobileNetV2-SSD-lite checkpoint can be downloaded from here:
  - https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
- Optimized checkpoint can be downloaded from the [Releases](/../../releases).


---

## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used as quantization scheme
- Cross-layer-Equalization and Adaround have been applied on optimized checkpoint
