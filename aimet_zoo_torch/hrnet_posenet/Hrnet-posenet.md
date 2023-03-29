# PyTorch HRNET-Posenet
This document describes evaluation of optimized checkpoint for Hrnet-posenet

## Environment Setup

### Get AIMET Model Zoo
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`

Add aimet_zoo_torch to your PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path to model zoo root>/aimet-model-zoo/
```

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.

### Install Dependencies
```bash
sudo -H pip install yacs
sudo -H pip install json-tricks
sudo -H pip install pycocotools
sudo -H pip install Cython
sudo -H pip install opencv-python==3.4.1.15
sudo -H pip install numpy==1.23
sudo -H apt-get update
sudo -H apt-get install ffmpeg libgl1
```

### Dataset
- This evaluation script is built to evaluate on COCO2017 validation images with person keypoints. 
- COCO dataset can be downloaded from here:
  - [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
  - [COCO 2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- The COCO dataset path should include coco images and annotations. It assumes a folder structure containing two subdirectories: `images/val2017` and `annotations`. Corresponding images and annotations should be put into the two subdirectories.

---

## Usage
```bash
python aimet_zoo_torch/hrnet_posenet/evaluators/hrnet_posenet_quanteval.py
	--model-config <configuration to be tested> \
	--dataset-path <path to MS-COCO validation dataset> \
	--use-cuda <boolean for using cuda, defaults to True>
```

Available model configurations are:
- hrnet_posenet_w4a8
- hrnet_posenet_w8a8

---

## Model checkpoints and configuration
- Downloading checkpoints and Quantization configuration file are handled through evaluation script.
- FP32 and Optimized checkpoint of HRNET-posenet can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

---


## Quantization Configuration
W8A8 | The following configuration has been used for quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 320 images (10 batches) from the validation dataloader was used for compute encodings
- Batchnorm folding and "TF" quantscheme in per channel mode has been applied to get the INT8 optimized checkpoint

W4A8 | The following configuration has been used for quantization:
- Weight quantization: 4 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 320 images (10 batches) from the validation dataloader was used for compute encodings
- Batchnorm folding and "TF" quantscheme in per channel mode has been applied to get the INT4 optimized checkpoint
