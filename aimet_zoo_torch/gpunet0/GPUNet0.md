# PyTorch GPUNet-0
This document describes evaluation of optimized checkpoints for GPUNet-0.

## Environment Setup

### Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.0.

### Add AIMET Model Zoo to the python path 
```bash 
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```

### Package Dependencies
Install required packages
```bash
pip install -r <path to aimet-model-zoo>/aimet_zoo_torch/gpunet0/requirements.txt
```

### Dataset
ImageNet can be downloaded from here:
  - https://image-net.org/download-images

The folder structure and format of ImageNet dataset is like below:
```
--ImageNet
	--val
	    --n01440764
	        --ILSVRC2012_val_00048969.JPEG
	--train
		--n13133613
		    --n13133613_7875.JPEG
```

---

## Usage
```bash
python gpunet0_quanteval.py \
		--dataset-path <The path to the ImageNet dataset's root path>
		--model-config <Quantized Model Configuration to test, default is 'gpunet0_w8a8', and just one choice>
		--batch-size <Data batch size to evaluate your model, default is 200>
		--use-cuda <Use cuda or cpu, default is True> \
```
* example 
    ```
    python gpunet0_quanteval.py --dataset-path <ILSVRC2012_PyTorch_path> --model-config gpunet0_w8a8
    ```
---

## Model checkpoint and configuration

- Optimized w8a8 checkpoint can be downloaded from here: [gpunet0_w8a8_checkpoint.pth](/../../releases/download/torch_gpunet0_w8a8/gpunet0_w8a8_checkpoint.pth)
- Optimized w8a8 encodings can be downloaded from here: [gpunet0_w8a8_torch.encodings](/../../releases/download/torch_gpunet0_w8a8/gpunet0_w8a8_torch.encodings)
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.23.0/user_guide/quantization_configuration.html) for more information on this file).

---

## Quantization Configuration (W8A8)
- Weight quantization: 8 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Percentile was used as quantization scheme, and the value was set to 99.999
- Adaround and fold_all_batch_norms_to_scale have been applied 

---

## Results
Below are the *acc top1* results of this GPUNet-0 implementation on ImageNet:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>acc top1 (%)</th>
  </tr>
  <tr>
    <td>GPUNet0_FP32</td>
    <td>78.86</td>
  </tr>
  <tr>
    <td>GPUNet0_FP32 + simple PTQ(w8a8)</td>
    <td>76.87</td>
  </tr>
  <tr>
    <td>GPUNet0_W8A8</td>
    <td>78.42</td>
  </tr>

</table>
