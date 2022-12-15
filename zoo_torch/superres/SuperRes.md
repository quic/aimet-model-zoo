# Super Resolution Family of Models
This document describes how to run AIMET quantization on the following model(s) and verify the performance of the quantized models. Example code is provided in the form of a [Jupyter Notebook](../examples/superres/notebooks/superres_quanteval.ipynb).
- QuickSRNet
- Anchor-based Plain Net (ABPN)
- Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution (XLSR)
- Super-Efficient Super Resolution (SESR)


## Environment setup
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) (*Torch GPU* variant) before proceeding further.

---
**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.23.0*  (i.e. set `release_tag="1.23.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).
---

## Install additional dependencies
To run the Jupyter Notebook, the following packages to be installed:
```
pip install jupyter==1.0.0
pip install notebook==6.4.3
```
When you install these packages, you should see a WARNING at the bottom, specifying to add a path to the environment variables. To do this, simply export the given path as:  
`export PATH=<path/from/warning/message>:$PATH`

## Download dataset
Download the *Set14 Super Resolution Dataset* from here: https://deepai.org/dataset/set14-super-resolution to any location in your workspace.

Extract the *set5.zip* file, then the *SR_testing_datasets.zip* found within.

The images of interest are located in the following path:  
`<root-path>/set5/SR_testing_datasets/Set14`

## Start Jupyter Notebook
Change to the directory containing the example code as follows:
`cd zoo_torch/examples`

Start the notebook server as follows (please customize the command line options using the help/documentation if appropriate):  
`jupyter notebook --ip=* --no-browser &`  
The above command will generate and display a URL in the terminal. Copy and paste it into your browser.

## Run Jupyter Notebook
Browse to the notebook at `superres` >> `notebooks` >> `superres_quanteval.ipynb`, and follow the instructions therein to run the code.

---
**Edit Jupyter Notebook**  
Please *replace the placeholders* within the Jupyter Notebook (such as the model checkpoint and the dataset paths) to point to the appropriate paths in your workspace into where you downloaded and extracted them.

---

## Models
Model checkpoints are available in the [Releases](/../../releases) page.

Please note the following regarding the available checkpoints:
- All model architectures were reimplemented from scratch and trained on the *DIV2k* dataset (available [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/)).
- The *float32* model is the full-precision model with the highest validation accuracy on the DIV2k dataset.
- The *int8* model is the quantized model with the highest validation accuracy obtained using AIMET's [Quantization-aware Training](https://developer.qualcomm.com/blog/exploring-aimet-s-quantization-aware-training-functionality).
- The above quantized model along with the Encodings were exported using AIMET's export tool.


## Quantization Configuration
In the evaluation notebook included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: *8 bits, per tensor symmetric quantization*
- Bias parameters are not quantized
- Activation quantization: *8 bits, asymmetric quantization*
- Model inputs are quantized
- *TF_enhanced* was used as the quantization scheme

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
    <td rowspan="2">ABPN</td>
    <td>N/A</td>
    <td>28</td>
    <td>32.71</td>
    <td>32.64</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>32</td>
    <td>32.75</td>
    <td>32.69</td>
  </tr>
  <tr>
    <td>XLSR</td>
    <td>N/A</td>
    <td>32</td>
    <td>32.57</td>
    <td>32.30</td>
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
  <tr>
    <td rowspan="3">QuickSRNet</td>
    <td>Small</td>
    <td>32</td>
    <td>32.52</td>
    <td>32.49</td>
  </tr>
  <tr>
    <td>Medium</td>
    <td>32</td>
    <td>32.78</td>
    <td>32.73</td>
  </tr>
  <tr>
    <td>Large</td>
    <td>64</td>
    <td>33.24</td>
    <td>33.17</td>
  </tr>
</table>

*<sup>[1]</sup>* Config: This parameter denotes a model configuration corresponding to a certain number of residual blocks used. The M*x* models have 16 feature channels, whereas the XL model has 32 feature channels.
