# Pytorch HRNet-W48

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.21/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.21.0.

## Experiment setup
- Get [HRNet pytorch-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1) branch:
  ```bash
  git clone https://github.com/HRNet/HRNet-Semantic-Segmentation.git -b pytorch-v1.1
  ```
- Makes sure additional dependencies [pyyaml](https://pyyaml.org/), [yacs](https://github.com/rbgirshick/yacs) are installed: 
  ```bash
  pip install pyyaml
  pip install 'yacs>=0.1.5'
  ```
- Add AIMET Model Zoo and the HRNet Lib to your pythonpath
```bash
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
export PYTHONPATH=$PYTHONPATH:<path to HRNet-Semantic-Segmentation>/lib
```

## Model checkpoints and configuration
- The original HRNet-W48 checkpoint can be downloaded from links provided at [HRNet pytorch-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1).
- Optimized HRNet checkpoint can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).
- Downloading optimized checkpoints and quantization configuration file are also handled through evaluation script.

## Dataset
- This evaluation was designed for Cityscapes dataset, which can be downloaded through registration on https://www.cityscapes-dataset.com/.
- After registration, go to https://www.cityscapes-dataset.com/downloads/ to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip
- Copy leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip to $SEG_ROOT/data/cityscapes/ ($SEG_ROOT denotes path to HRNet-Semantic-Segmentation)

<strong> NOTE! Data has to be organized in the following way: </strong>

```
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
│   ├── lip
│   │   ├── testvalList.txt
│   │   ├── trainList.txt
│   │   └── valList.txt
```

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python hrnet-w48_quanteval.py \
	--default-param-bw <8|4> \
	--hrnet-path <Direct path way to HRnet github repo locally> \
	--use-cuda <Use GPU for evaluation>

```

## Quantization Configuration
- Weight quantization: 8 or 4 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used for weight quantization scheme
- TF_enhanced was used for activation quantization scheme
- Cross layer equalization and Adaround have been applied on optimized checkpoint
- 2K Images from Cityscapes test dataset are used as calibration dataset
