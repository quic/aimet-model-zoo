# PyTorch Classification models
This document describes evaluation of optimized checkpoints for Resnet18, Resnet50 and Regnet_x_3_2gf.

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Additional Setup Dependencies
```
sudo -H pip install torchvision==0.11.2 --no-deps
sudo -H chmod 777 -R <path_to_python_package>/dist-packages/*
```

## Obtain the Original Model for Comparison
- [Pytorch Torchvision hub](https://pytorch.org/vision/0.11/models.html#classification) instances of Resnet18, Resnet50 and Regnet_x_3_2gf are used as reference FP32 models. These instances are optimized using AIMET to obtain quantized optimized checkpoints.

## Experiment setup
```python
export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo
```

## Dataset
This evaluation was designed for the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012), which can be obtained from: http://www.image-net.org/  
The dataset directory is expected to have 3 subdirectories: train, valid, and test (only the valid test is used, hence if the other subdirectories are missing that is ok).
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

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
cd classification
python classification_quanteval.py\
	--fp32-model <name of the fp32 torchvision model - resnet18/resnet50/regnet_x_3_2gf> \
	--default-param-bw <weight bitwidth for quantization - 8 for INT8, 4 for INT4> \
	--default-output-bw <output bitwidth for quantization - 8 for INT8> \
	--use-cuda <boolean for using cuda> \
	--evaluation-dataset <path to Imagenet validation dataset>

# Example
python classification_quanteval.py --fp32-model=resnet18 --default-weight-bw=8 --default-output-bw=8 --use-cuda=True --evaluation-dataset=<path_to_ImageNet_val_dataset>
```

## Quantization Configuration
INT8 optimization

The following configuration has been used for the above models for INT8 quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 2000 images from the calibration dataset were used for computing encodings
- TF_enhanced was used as quantization scheme
- Cross layer equalization and Adaround in per channel mode has been applied for all the models to get the best INT8 optimized checkpoint

INT4 optimization

The following configuration has been used for the above models for INT4 quantization:
- Weight quantization: 4 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 2000 images from the calibration dataset were used for computing encodings
- TF_enhanced was used as quantization scheme
- Cross layer equalization and Adaround in per channel mode has been applied for all the models to get the best INT4 optimized checkpoint
