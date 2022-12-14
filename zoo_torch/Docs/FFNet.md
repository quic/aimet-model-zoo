# PyTorch-FFNet

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Additional Dependencies
1. Install skimage as follows
```
pip install scikit-image
```

## Model modifications & Experiment Setup
1. Clone the [FFNet repo](https://github.com/Qualcomm-AI-research/FFNet)
```
git clone https://github.com/Qualcomm-AI-research/FFNet
```
2. Hardcopy the two folders below for dataloader and model evaluation imports 
```
datasets/cityscapes/dataloader
datasets/cityscapes/utils
```
3. Add AIMET Model Zoo to the pythonpath 
```bash 
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```

## Dataset
The Cityscape Dataset can be downloaded from here:
  - https://www.cityscapes-dataset.com/

In the `datasets/cityscapes/dataloader/base_loader.py` script, change the Cityscape dataset path to point to the path where the dataset was downloaded.

## Model checkpoint and configuration

- The original prepared FFNet checkpoint can be downloaded from here:
  - https://github.com/quic/aimet-model-zoo/releases/tag/torch_segmentation_ffnet
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/release-aimet-1.22/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.22.2/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python ffnet_quanteval.py \
		--model-name <model name for quantization, default is segmentation_ffnet78S_dBBB_mobile> \
		--use-cuda <Use cuda or cpu, default is True> \
		--batch-size <Number of images per batch, default is 8>
```

## Quantization Configuration (INT8)
- Weight quantization: 8 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization (CLE) has been applied on optimized checkpoint
- for low resolution models with pre_down suffix, the GaussianConv2D layer is disabled for quantization.

## Results
Below are the *mIoU* results of the PyTorch FFNet model for the Cityscapes dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>FP32 (%)</th>
    <th>INT8 (%)</th>
  </tr>
  <tr>
    <td>segmentation_ffnet78S_dBBB_mobile</td>
    <td>81.3</td>
    <td>80.7</td>
  </tr>
  <tr>
    <td>segmentation_ffnet54S_dBBB_mobile</td>
    <td>80.8</td>
    <td>80.1</td>
  </tr>
  <tr>
    <td>segmentation_ffnet40S_dBBB_mobile</td>
    <td>79.2</td>
    <td>78.9</td>
  </tr>
  <tr>
    <td>segmentation_ffnet78S_BCC_mobile_pre_down</td>
    <td>80.6</td>
    <td>80.4</td>
  </tr>
  <tr>
    <td>segmentation_ffnet122NS_CCC_mobile_pre_down</td>
    <td>79.3</td>
    <td>79.0</td>
  </tr>
</table>
