# Pytorch DeepLabV3+ (Semantic Segmentation)

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.23.

### Install dependencies 
```bash 
   python -m pip install pycocotools
```
Append the repo location to your `PYTHONPATH` with the following:  
  ```bash
  export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo>
  ```

### Dataset 
The Pascal Dataset can be downloaded from here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/


---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python3  aimet_zoo_torch/deeplabv3/evaluators/deeplabv3_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to the downloaded Pascal dataset, should end in VOCdevkit/VOC2012> \
                --batch-size  <batch size as an integer value, defaults to 8> \
```

Available model configurations are:
- dlv3_w4a8
- dlv3_w8a8

---


## Model checkpoint and configuration

- The original DeepLabV3+ checkpoint can be downloaded from here:
  - https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).


---

## Quantization Configuration
W8A8 Quantization | The following configuration has been used:
- Weight quantization: 8 bits, per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround has been applied on optimized checkpoint
- Data Free Quantization has been performed on the optimized checkpoint

W4A8 Quantization | The following configuration has been used:
- Weight quantization: 4 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround has been applied on optimized checkpoint
- Data Free Quantization has been performed on the optimized checkpoint
- Quantization Aware Traning has been performed on the optimized checkpoint
