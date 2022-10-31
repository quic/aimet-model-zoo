# Pytorch InverseForm (Semantic Segmentation)

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Experiment setup
- Clone the [InverseForm](https://github.com/Qualcomm-AI-research/InverseForm) repo and apply patch
```bash
  git clone https://github.com/Qualcomm-AI-research/InverseForm.git
  cd InverseForm
  git checkout d5fec5b065c9a3c7afab48db84071fc3537bad7a
  patch -p1 < ../aimet-model-zoo/zoo_torch/examples/inverseform/inverseform_eval.patch
```
- Install dependencies 
```bash 
   python -m pip install runx
   python -m pip install fire
   python -m pip install scikit-image
```
- Append the repo location to your `PYTHONPATH` with the following:
  `export PYTHONPATH=<path to inverseform repo>:$PYTHONPATH`

## Model checkpoints and configuration
- The InverseForm model checkpoints can be downloaded from [releases](/../../releases/tag/inverseform).

## Dataset
- Benchmark dataset can be downloaded from here:
  - [Cityscapes](https://www.cityscapes-dataset.com/)
- Follow [Cityscapes path](https://github.com/Qualcomm-AI-research/InverseForm#cityscapes-path) instructions.

## Usage
- To run HRNet-16-Slim-IF evaluation with QuantSim in AIMET, use the following
```bash
python inverseform_quanteval.py \
	--checkpoint-prefix inverseform-w16_w8a8 \
	--arch "lighthrnet.HRNet16" --hrnet_base "16" \
	--use-cuda <Run evaluation on GPU>
```
- To run OCRNet-48-IF evaluation with QuantSim in AIMET, use the following
```bash
python inverseform_quanteval.py \
	--checkpoint-prefix inverseform-w48_w8a8 \
	--arch "ocrnet.HRNet" --hrnet_base "48" \
	--use-cuda <Run evaluation on GPU>
```

## Quantization configuration
- Weight quantization: 8 bits per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround have been applied on optimized checkpoint
