# PyTorch-DeepLabV3+

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies
1. Install pycocotools as follows
```
sudo -H pip install pycocotools
```

## Model modifications & Experiment Setup
1. Clone the [DeepLabV3+ repo](https://github.com/jfzhang95/pytorch-deeplab-xception)
```
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd pytorch-deeplab-xception
git checkout 9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6
```
2. Apply the following patch to the above repository
```
git apply ../aimet-model-zoo/zoo_torch/examples/pytorch-deeplab-xception-zoo.patch
```
3. Place modeling directory & dataloaders directory & metrics.py & mypath.py to aimet-model-zoo/zoo_torch/examples/
```
mv modeling ../aimet-model-zoo/zoo_torch/examples/
mv dataloaders ../aimet-model-zoo/zoo_torch/examples/
mv utils/metrics.py ../aimet-model-zoo/zoo_torch/examples/
mv mypath.py ../aimet-model-zoo/zoo_torch/examples/
```
4. Download Optimized DeepLabV3+ checkpoint from [Releases](/../../releases).
5. Change data location as located in mypath.py

## Obtaining model checkpoint and dataset

- The original DeepLabV3+ checkpoint can be downloaded here:
  - https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view
- Optimized DeepLabV3+ checkpoint can be downloaded from  [Releases](/../../releases).
- Pascal Dataset can be downloaded here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

## Usage

- To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_deeplabv3.py \
        --checkpoint-path   <path to optimized checkpoint directory to load from> \
        --base-size         <base size for Random Crop> \
        --crop-size         <crop size for Random Crop> \
        --num-classes       <number of classes in a dataset> \
        --dataset           <dataset to be used for evaluation> \
        --quant-scheme      <quantization schme to run> \
        --default-output-bw <bitwidth for activation quantization> \
        --default-param-bw  <bitwidth for weight quantization>     		
```

## Quantization Configuration
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Data Free Quantization and Quantization aware Training has been performed on the optimized checkpoint
