# PyTorch SegNet
This document describes evaluation of optimized checkpoints for SegNet

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.23.

### Additional Setup Dependencies
```bash
    pip install scikit-image
```

### Experiment setup
- Clone the [aimet-model-zoo](https://github.com/quic/aimet-model-zoo.git) repo
```bash
    git clone https://github.com/quic/aimet-model-zoo.git
```
- Append the repo location to your `PYTHONPATH` with the following:
```bash
    export PYTHONPATH=$PYTHONPATH:<path to aimet_model_zoo_path>
```

### Dataset
This evaluation was designed for the CamVid dataset variant provided by SegNet authors [repository](https://github.com/alexgkendall/SegNet-Tutorial).
- Download and extract CamVid directory:
```bash
    wget https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip
    unzip master.zip && mv SegNet-Tutorial-master/CamVid . && rm -r SegNet-Tutorial-master
```

### Model checkpoints and configuration
- The SegNet model checkpoints can be downloaded from the [Releases](/../../releases) page.

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
    python3 aimet-model-zoo/aimet_zoo_torch/segnet/evaluator/segnet_quanteval.py \
        --dataset-path <path to CamVid dataset> \
        --model-config <configuration to be tested>
```

Available model configurations are:
- segnet_w8a8
- segnet_w4a8

---

## Quantization Configuration
- Weight quantization: 8 or 4 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used as quantization scheme
