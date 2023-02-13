# Pytorch InverseForm (Semantic Segmentation)

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.23.

### Install dependencies 
```bash 
   python -m pip install runx
   python -m pip install fire
   python -m pip install scikit-image
```

### Additional setup 
Append the repo location to your `PYTHONPATH` with the following:  
  ```bash
  export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo>
  ```

### Dataset
Benchmark dataset can be downloaded from here: 
- [Cityscapes](https://www.cityscapes-dataset.com/)

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python3  aimet_zoo_torch/inverseform/evaluators/inverseform_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to directory containing CityScapes> \
                --batch-size  <batch size as an integer value, defaults to 2> \
```

Available model configurations are:
- hrnet_16_slim_if
- ocrnet_48_if

---

## Model checkpoints and configuration
- The InverseForm model checkpoints can be downloaded from the [Releases](/../../releases) page.

---

## Quantization configuration
- Weight quantization: 8 bits per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround have been applied on optimized checkpoint
