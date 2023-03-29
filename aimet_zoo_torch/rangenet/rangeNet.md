# PyTorch RangeNet++

## Environment Setup

### Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.0.

### Add AIMET Model Zoo to the python path 
```bash 
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```

### Package Dependencies
Install required packages
```bash
pip install -r <path to aimet-model-zoo>/aimet_zoo_torch/rangenet/requirements.txt
```

### Dataset
Semantic Kitti can be downloaded from here:
  - http://www.semantic-kitti.org/dataset.html

The folder structure and format of Semantic kitti dataset is like below:
```
--dataset
	--sequences
		--00
			--velodyne
				--000000.bin
				--000001.bin
			--labels
				--000000.label
				--000001.label
			--poses.txt
```

---

## Usage
```bash
python rangenet_quanteval.py \
		--dataset-path <The path to the dataset, default is '../models/train/tasks/semantic/dataset/'>
		--use-cuda <Use cuda or cpu, default is True> \
```

---

## Model checkpoint and configuration

- The original prepared RangeNet++ checkpoint can be downloaded from here: [darknet21.tar.gz](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet21.tar.gz) or [rangeNet_plus_FP32.tar.gz](/../../releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_FP32.tar.gz)
- Optimized w8a8 checkpoint can be downloaded from here: [rangeNet_plus_w8a8_checkpoint.pth](/../../releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_w8a8_checkpoint.pth)
- Optimized w4a8 checkpoint can be downloaded from here: [rangeNet_plus_w4a8_checkpoint.pth](/../../releases/download/torch_rangenet_plus_w4a8/rangeNet_plus_w4a8_checkpoint.pth)
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.23.0/user_guide/quantization_configuration.html) for more information on this file).

---

## Quantization Configuration (W4A8/W8A8)
- Weight quantization: 4 bits for w4a8, 8 bits for w8a8, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Percentile was used as quantization scheme, and the value was set to 99.99
- Bn fold and Adaround have been applied on optimized checkpoint

---

## Results
Below are the *mIoU* results of this RangeNet++ implementation on SemanticKitti:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>mIoU (%)</th>
  </tr>
  <tr>
    <td>rangeNet_plus_FP32</td>
    <td>47.2</td>
  </tr>
  <tr>
    <td>rangeNet_plus_FP32 + simple PTQ(w8a8)</td>
    <td>45.0</td>
  </tr>
  <tr>
    <td>rangeNet_plus_W8A8_checkpoint</td>
    <td>47.1</td>
  </tr>
  <tr>
    <td>rangeNet_plus_W4A8_checkpoint</td>
    <td>46.8</td>
  </tr>
</table>
