# PyTorch MobileNetV2

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Obtain the Original Model for Comparison
1. Clone the [MobileNetV2 repo](https://github.com/tonylins/pytorch-mobilenet-v2)
```
git clone https://github.com/tonylins/pytorch-mobilenet-v2
cd pytorch-mobilenet-v2/
git checkout 99f213657e97de463c11c9e0eaca3bda598e8b3f
```
2. Place model definition under model directory
```
mkdir ../aimet-model-zoo/zoo_torch/examples/mobilenetv2/model
mv MobileNetV2.py ../aimet-model-zoo/zoo_torch/examples/mobilenetv2/model
```

## Add AIMET Model Zoo to the pythonpath
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
To run evaluation with QuantSim in AIMET, use the following:
```bash
python mobilenetv2_quanteval.py \
	--dataset-path <path to imagenet root directory> \
	--batch-size <batch size as an integer value>  
```


## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Data Free Quantization and Quantization aware Training has been performed on the optimized checkpoint
