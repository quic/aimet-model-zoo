# Pytorch HRNet-W48

## Environment setup

### Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.0.

- Add AIMET Model Zoo and the HRNet Lib to your pythonpath
```bash
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```

### Dependencies
- Makes sure additional dependencies [pyyaml](https://pyyaml.org/), [yacs](https://github.com/rbgirshick/yacs) are installed: 
  ```bash
  pip install pyyaml
  pip install 'yacs>=0.1.5'
  ```


### Dataset
- This evaluation was designed for Cityscapes dataset, which can be downloaded through registration on https://www.cityscapes-dataset.com/.
- After registration, go to https://www.cityscapes-dataset.com/downloads/ to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip


### Model checkpoints and configuration
- The original HRNet-W48 checkpoint can be downloaded from links provided at [HRNet pytorch-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).
- Optimized HRNet checkpoint can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).
- Downloading optimized checkpoints and quantization configuration file are also handled through evaluation script.

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python hrnet_sem_seg_quanteval.py \
	--model-config <model configuation to be tested> \
	--dataset-path <path to cityscapes directory containing gtFine and leftImg8bit_trainvaltest subdirectories> \
  --use-cuda <whether to compute on GPU or CPU>

```

Available model configurations are:
- hrnet_sem_seg_w4a8
- hrnet_sem_seg_w8a8

---

## Quantization Configuration
- Weight quantization: 8 or 4 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used for weight quantization scheme
- TF_enhanced was used for activation quantization scheme
- Cross layer equalization and Adaround have been applied on optimized checkpoint
- 2K Images from Cityscapes test dataset are used as calibration dataset
