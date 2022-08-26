# Quantization evaluation: Classification models
This document describes evaluation of optimized checkpoints for Resnet18, Resnet50 and Regnet_x_3_2gf.


**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.21.0*  (i.e. set `release_tag="1.21.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```
sudo -H pip install torchvision==0.11.2 --no-deps
sudo -H chmod 777 -R <path_to_python_package>/dist-packages/*
```


## Obtaining model checkpoint, ImageNet validation dataset and calibration dataset


- <a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision hub</a> instances of Resnet18, Resnet50 and Regnet_x_3_2gf are used as refernce FP32 models. These instances are optimized using AIMET to obtain quantized optimized checkpoints.
- Optimized Resnet18, Resnet50 and Regnet_x_3_2gf checkpoint can be downloaded from the [Releases](/../../releases) page.
- ImageNet can be downloaded from here:
  - http://www.image-net.org/
- Use standard validation set of ImageNet dataset (50k images set) for evaluting performance of FP32 and quantized models.

For the quantization task, we require the model path, evaluation dataset path and calibration dataset path - which is a subset of validation dataset to be used for computing the encodings and AdaRound optimizaiton.


## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
cd classification
python classification_quanteval.py\
		--fp32-model <name of the fp32 torchvision model - resnet18/resnet50/regnet_x_3_2gf> \
		--default-param-bw <weight bitwidth for quantization - 8 for INT8>\
		--default-output-bw <output bitwidth for quantization - 8 for INT8>\
		--use-cuda <boolean for using cuda>  \
		--evaluation-dataset <path to Imagenet validation dataset> \
		
eg.
python classification_quanteval.py --fp32-model=resnet18 --default-weight-bw=8 --default-output-bw=8 --use-cuda=True --evaluation-dataset=<path_to_ImageNet_val_dataset>
```

## Quantization Configuration
The following configuration has been used for the above models for INT8 quantization

- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 2000 images from the calibration dataset were used for computing encodings
- TF_enhanced was used as quantization scheme
- Cross layer equalization and Adaround in per channel mode has been applied for all the models to get the best INT8 optimized checkpoint




