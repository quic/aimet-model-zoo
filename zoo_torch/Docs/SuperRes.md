# Super Resolution Family of Models
This document describes how to run AIMET quantization on the following model(s) and verify the performance of the quantized models. Example code is provided in the form of a [Jupyter Notebook](../examples/superres/notebooks/superres_quanteval.ipynb).
- Anchor-based Plain Net (ABPN)
- Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution (XLSR)
- Super-Efficient Super Resolution (SESR)


## Environment setup
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) (*Torch GPU* variant) before proceeding further.

---
**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.22.2*  (i.e. set `release_tag="1.22.2"` in the above instructions).
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
- Weight quantization: *8 bits, per tensor asymmetric quantization*
- Bias parameters are not quantized
- Activation quantization: *8 bits, asymmetric quantization*
- Model inputs are quantized
- *TF_enhanced* was used as the quantization scheme
