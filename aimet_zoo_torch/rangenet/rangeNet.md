# PyTorch RangeNet++

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.0.

## Model modifications & Experiment Setup
1. Clone the [RangeNet++ repo](https://github.com/PRBonn/lidar-bonnetal)
```
git clone https://github.com/PRBonn/lidar-bonnetal.git
```
2. Apply patches to darknet.py in the above repo using the command below:
```bash
patch /path/to/lidar-bonnetal/train/backbones/darknet.py /path/to/aimet-model-zoo/aimet_zoo_torch/rangenet/train/models/backbones/darknet.patch

path /path/to/lidar-bonnetal/train/tasks/semantic/decoders/darknet.py /path/to/aimet-model-zoo/aimet_zoo_torch/rangenet/train/tasks/semantic/decoders/darknet.patch
```
These changes are needed in order to meet prepare_model's requirements

2. Create a new folder to put your downloaded dataset

3. Create a new folder to put your downloaded original/optimized model

4. Add the "models/train/tasks/semantic/evaluate.py" file to your "models/train/tasks/semantic" path

5. Add AIMET Model Zoo to the python path 
```bash 
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```


## Dataset
The Semantic kitti Dataset can be downloaded from here:
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

## Model checkpoint and configuration

- The original prepared RangeNet++ checkpoint can be downloaded from here:
  - http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet21.tar.gz
  or
  - https://github.com/quic/aimet-model-zoo/releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_FP32.tar.gz
- Optimized checkpoint can be downloaded from the:
  - https://github.com/quic/aimet-model-zoo/releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_INT8_checkpoint.pth
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.23.0/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python rangenet++_quanteval.py \
		--dataset-path <The path to the dataset, default is '../models/train/tasks/semantic/dataset/'>
		--model-orig-path <The path to the model_orig, default is '../models/train/tasks/semantic/pre_trained_model'>
		--model-optim-path <The path to the model_optim, default is '../models/train/tasks/semantic/quantized_model'>
		--use-cuda <Use cuda or cpu, default is True> \
		--batch-size <Number of images per batch, default is 1>
```

## Quantization Configuration (W8A8)
- Weight quantization: 8 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Percentile was used as quantization scheme, and the value was set to 99.99
- Bn fold and Adaround have been applied on optimized checkpoint

## Results
Below are the *mIoU* results of the PyTorch rangeNet++ model for the semantic kitti dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>FP32 (%)</th>
    <th>W8A8 (%)</th>
  </tr>
  <tr>
    <td>rangeNet_plus_FP32</td>
    <td>47.2</td>
    <td>46.8</td>
  </tr>
  <tr>
    <td>rangeNet_plus_W8A8_checkpoint</td>
    <td> - </td>
    <td>47.0</td>
  </tr>
</table>
