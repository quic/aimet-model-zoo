# PyTorch HRNET-Posenet
This document describes evaluation of optimized checkpoint for Hrnet-posenet

## Workspace setup
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Additional Setup Dependencies
```bash
sudo -H pip install yacs
sudo -H pip install json-tricks
sudo -H pip install pycocotools
sudo -H pip install Cython
sudo -H pip install opencv-python==3.4.1.15
sudo -H pip install numpy==1.23
sudo -H apt-get update
sudo -H apt-get install ffmpeg libgl1
sudo -H chmod 777 -R <path_to_python_package>/dist-packages/*

cd <path_to_aimet_modelzoo>/aimet_zoo_torch
git clone https://github.com/HRNet/HRNet-Human-Pose-Estimation.git
cd HRNet-Human-Pose-Estimation/
git checkout 00d7bf72f56382165e504b10ff0dddb82dca6fd2
cp -r ./lib/ ../hrnet-posenet/

cd aimet_zoo_torch/hrnet-posenet/lib
make
```

Add aimet_zoo_torch/hrnet-posenet/lib to your PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path to model zoo root>
export PYTHONPATH=$PYTHONPATH:<path to root>/HRNet-Human-Pose-Estimation/lib
```

## Modifications
Add the following lines inside `./hrnet-posenet/lib/core/function.py`

Addition at line 105
```
on_cuda = next(model.parameters()).is_cuda
```

Addition at line 121
```
if on_cuda:
	input=input.cuda()
```

## Model checkpoints and configuration
- Downloading checkpoints and Quantization configuration file are handled through evaluation script.
- FP32 and Optimized checkpoint of HRNET-posenet can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Experiment setup
```python
export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo
```

## Dataset
- This evaluation script is built to evaluate on COCO2014 validation images with person keypoints. 
- COCO dataset can be downloaded from here:
  - [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip)
  - [COCO 2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- The COCO dataset path should include coco images and annotations. It assumes a folder structure containing two subdirectories: `images/val2014` and `annotations`. Corresponding images and annotations should be put into the two subdirectories.

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
cd <path_to_aimet_modelzoo>/aimet_zoo_torch/examples/hrnet-posenet
python hrnet_posenet_quanteval.py
	--default-param-bw <weight bitwidth for quantization - 8 for INT8, 4 for INT4> \
	--default-output-bw <output bitwidth for quantization - 8 for INT8> \
	--use-cuda <boolean for using cuda> \
	--evaluation-dataset <path to MS-COCO validation dataset>

eg.
python hrnet_posenet_quanteval.py --default-param-bw=8 --default-output-bw=8 --use-cuda=True --evaluation-dataset=<path_to_MSCOCO_mainDIR>
```

## Quantization Configuration
INT8 optimization

The following configuration has been used for the above model for INT8 quantization
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 320 images (10 batches) from the validation dataloader was used for compute encodings
- Batchnorm folding and "TF" quantscheme in per channel mode has been applied to get the INT8 optimized checkpoint

INT4 optimization

The following configuration has been used for the above model for INT4 quantization
- Weight quantization: 4 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 320 images (10 batches) from the validation dataloader was used for compute encodings
- Batchnorm folding and "TF" quantscheme in per channel mode has been applied to get the INT4 optimized checkpoint
