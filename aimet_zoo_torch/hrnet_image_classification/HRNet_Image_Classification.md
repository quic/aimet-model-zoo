# Pytorch HRNet-W32 for image classification 

## Environment setup

### Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.0.

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

## Dataset
This evaluation was designed for the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012), which can be obtained from: http://www.image-net.org/  
The dataset directory is expected to have 3 subdirectories: train, valid, and test (only validation dataset is needed for this evaluation).
Each of the {train, valid, test} directories is then expected to have 1000 subdirectories, each containing the images from the 1000 classes present in the ILSVRC2012 dataset, such as in the example below:

```
  train/
  ├── n01440764
  │   ├── n01440764_10026.JPEG
  │   ├── n01440764_10027.JPEG
  │   ├── ......
  ├── ......
  val/
  ├── n01440764
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   ├── ......
  ├── ......
```
---

### Model checkpoints and configuration
- The original HRNet-W32 checkpoint can be downloaded from links provided at [HRNet-W32-C](https://github.com/HRNet/HRNet-Image-Classification).
- Optimized HRNet checkpoint can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).
- Downloading optimized checkpoints and quantization configuration file are also handled through evaluation script.

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python hrnet_image_classification_quanteval.py \
	--model-config <model configuation to be tested> \
	--dataset-path <path imagenet dataset> \
        --use-cuda <whether to compute on GPU or CPU>

```

Available model configurations are:
- hrnet_w32_w8a8

---

## Quantization Configuration
- Weight quantization: 8 bits per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used for quantization scheme
- Autoquant have been applied on optimized checkpoint
- 2 images per class (1000 classes) from ImageNet dataset are used as calibration dataset
