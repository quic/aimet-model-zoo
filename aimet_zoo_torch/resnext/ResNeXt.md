# PyTorch ResNeXt (Image Classification)
This document describes evaluation of optimized checkpoints for ResNeXt 

## Environment Setup
### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.

### Additional Setup Dependencies
```
pip install torchvision==0.11.2 --no-deps
```

### Obtain the Original Model for Comparison
- [Pytorch Torchvision hub](https://pytorch.org/vision/0.11/models.html#classification) instances of ResNeXt101_32x8d are used as reference FP32 models. These instances are optimized using AIMET to obtain quantized optimized checkpoints.

### Experiment setup
```
git clone https://github.com/quic/aimet-model-zoo.git
```
```python
export PYTHONPATH=$PYTHONPATH:<path to aimet_model_zoo_path>
```

## Dataset
This evaluation was designed for the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012), which can be obtained from: http://www.image-net.org/  
The dataset directory is expected to have 3 subdirectories: train, valid, and test (only the validation dataset is used for this evaluation).
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

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python resnext_quanteval.py\
  --model-config <configuration to be tested> \
  --dataset-path <path to ImageNet dataset> \
  --use-cuda <whether to run on GPU or cpu>
```

Available model configurations are:
- resnext101_w8a8

---

## Quantization Configuration

The following configuration has been used for the above models for INT8 quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 2 images per class (2000 total images) from the calibration dataset were used for computing encodings
- TF_enhanced was used as quantization scheme
- Batch norm folding was applied to get the best W8A8 optimized checkpoint

