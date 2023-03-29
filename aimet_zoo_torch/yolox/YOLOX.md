# PyTorch-YOLOX

## Environment Setup

### Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.

### Additional Dependencies
Install necessary dependencies as follows:
```
pip install pycocotools
```

### Add AIMET Model Zoo to the pythonpath
```bash 
export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo_path>
```

### Dataset
The MS-COCO 2017 validation dataset can be downloaded from here:
  - http://images.cocodataset.org/zips/val2017.zip

The dataset's root folder (which you pass an as arg), should have 2 subfolders: annotations, and images. The annotations folder should contain only json files. The images folder should contain three subfolders: train2017, val2017, test2017.

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python3 aimet_zoo_torch/yolox/evaluators/yolox_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <Path to MS-COCO 2017> \
                --batch-size <Number of images per batch, default is 64>
```
Available model configurations are:
- yolox_s
- yolox_l
---

## Model checkpoint and configuration

- The original prepared YOLOX checkpoint can be downloaded from here:
  - https://github.com/quic/aimet-model-zoo/releases/download/torch_yolox_int8
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/release-aimet-1.25/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.25.0/user_guide/quantization_configuration.html) for more information on this file).

---

## Quantization Configuration (INT8)
- Weight quantization: 8 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Percentile was used as quantization scheme
  - percentile value is set as 99.9942 by searching for YOLOX-s
  - percentile value is set as 99.99608 by searching for YOLOX-l
- BatchNorm Folding (BNF) has been applied on optimized checkpoint

## Results
Below are the *mAP@0.50:0.95* results of the PyTorch YOLOX model for the MS-COCO2017 dataset:

<table style= " width:50%">
  <tr>
    <th>Model Configuration</th>
    <th>FP32 (%)</th>
    <th>INT8 (%)</th>
  </tr>
  <tr>
    <td>YOLOX-s</td>
    <td>40.5</td>
    <td>39.7</td>
  </tr>
  <tr>
    <td>YOLOX-l</td>
    <td>49.7</td>
    <td>48.8</td>
  </tr>
</table>

