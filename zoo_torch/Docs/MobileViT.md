# PyTorch Transformer model Mobile Vision Transformer(MobileViT) for Image Classification 
This document describes evaluation of optimized checkpoints for mobile vision transformer (MobileViT) for image classification 

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) (*Torch GPU* variant) before proceeding further.

**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.23.0*  (i.e. set `release_tag="1.23.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```
pip install accelerate==0.9.0
pip install transformers==4.21.0
pip install datasets==2.4.0

```

## Model checkpoint
- Original full precision checkpoints without downstream training were downloaded through hugging face 
- [Full precision model with downstream training weight files] are automatically downloaded using evaluation script 
- [Quantization optimized model weight files] are automatically downloaded using evaluation script 


## Dataset
- - This evaluation was designed for the [2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012)](http://www.image-net.org/). The dataset directory is expected to have 3 subdirectories: train, valid, and test (only the valid test is used, hence if the other subdirectories are missing that is ok).
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
- To run evaluation with QuantSim in AIMET, use the following
```bash

python transformer_imageclassification.py \
    --model_type <model_type> \
    --model_eval_type <model_eval_type> \
    --train_dir <imagenet_train_path> \
    --validation_dir <imagenet_val_path> \
    --per_device_eval_batch_size <batch_size>


# Example
python transformer_imageclassification.py --model_name_or_path mobilevit  --model_eval_type fp32  --per_device_eval_batch_size 4 --train_dir <imagenet_train_path> --validation_dir <imagenet_val_path> --per_device_eval_batch_size 8 
```

- supported keywords of model_type are vit and mobilevit  
- supported keywords of model_eval_type are int8 and fp32 

## Quantization Configuration
The following configuration has been used for the above models for INT8 quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF range learning  was used as quantization scheme
- Clamped initialization was adopted
- Quantization aware training (QAT) was used to obtain optimized quantized weights

