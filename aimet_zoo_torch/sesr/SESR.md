# Super Efficient Super Resolution (SESR)
This document describes how to run AIMET quantization on the following model(s) and verify the performance of the quantized models. 

---

## Environment Setup

### 1. Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further. This model was tested with the `torch_gpu` variant of AIMET version 1.24.0.

### 2. Install AIMET-Model-Zoo
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`  
`export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo>`  

### 3. Download dataset
Download the *Set14 Super Resolution Dataset* from here: https://deepai.org/dataset/set14-super-resolution to any location in your workspace.

Extract the *set5.zip* file, then the *SR_testing_datasets.zip* found within.

The images of interest are located in the following path:  
`<root-path>/set5/SR_testing_datasets/Set14`

---

## Running Evaluation

To run evaluation with QuantSim in AIMET, use the following
```bash
 python3  sesr_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to directory containing High Resolution (ground truth) images>
```

Available model configurations are:
- sesr_m3_2x_w8a8
- sesr_m3_3x_w8a8
- sesr_m3_4x_w8a8
- sesr_m5_2x_w8a8
- sesr_m5_3x_w8a8
- sesr_m5_4x_w8a8
- sesr_m7_2x_w8a8
- sesr_m7_3x_w8a8
- sesr_m7_4x_w8a8
- sesr_m11_2x_w8a8
- sesr_m11_3x_w8a8
- sesr_m11_4x_w8a8
- sesr_xl_2x_w8a8
- sesr_xl_3x_w8a8
- sesr_xl_4x_w8a8


---

## Models
Model checkpoints are available in the [Releases](/../../releases) page.

Please note the following regarding the available checkpoints:
- All model architectures were reimplemented from scratch and trained on the *DIV2k* dataset (available [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/)).
- The *float32* model is the full-precision model with the highest validation accuracy on the DIV2k dataset.
- The *int8* model is the quantized model with the highest validation accuracy obtained using AIMET's [Quantization-aware Training](https://developer.qualcomm.com/blog/exploring-aimet-s-quantization-aware-training-functionality).
- The above quantized model along with the Encodings were exported using AIMET's export tool.

---

## Quantization Configuration
In the evaluation notebook included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: *8 bits, per tensor symmetric quantization*
- Bias parameters are not quantized
- Activation quantization: *8 bits, asymmetric quantization*
- Model inputs are quantized
- *TF_enhanced* was used as the quantization scheme

---

## Results
**NOTE:**
All results below used a *Scaling factor (LR-to-HR upscaling) of 2x* and the *Set14 dataset*.
<table style= " width:50%">
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Config<sup>[1]</sup></th>
    <th rowspan="2">Channels</th>
    <th colspan="2" style="text-align:center;">PSNR</th>
  </tr>
  <tr>
    <th>FP32</td>
    <th>INT8</td>
  </tr>
  <tr>
    <td rowspan="5">SESR</td>
    <td>M3</td>
    <td>16</td>
    <td>32.41</td>
    <td>32.25</td>
  </tr>
  <tr>
    <td>M5</td>
    <td>16</td>
    <td>32.57</td>
    <td>32.50</td>
  </tr>
  <tr>
    <td>M7</td>
    <td>16</td>
    <td>32.66</td>
    <td>32.58</td>
  </tr>
  <tr>
    <td>M11</td>
    <td>16</td>
    <td>32.73</td>
    <td>32.59</td>
  </tr>
  <tr>
    <td>XL</td>
    <td>32</td>
    <td>33.03</td>
    <td>32.92</td>
  </tr>
</table>

*<sup>[1]</sup>* Config: This parameter denotes a model configuration corresponding to a certain number of residual blocks used. The M*x* models have 16 feature channels, whereas the XL model has 32 feature channels.
