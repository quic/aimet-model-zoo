# PyTorch SSD_Res50

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.

### Install dependencies 
```bash 
   python -m pip install pycocotools gdown
```
We also need the original source as dependency for data processing purpose. You can clone the repo using the command:
```bash
   git clone https://github.com/uvipen/SSD-pytorch.git
```
Append the repo location to your `PYTHONPATH` with the following:  
  ```bash
  export PYTHONPATH=$PYTHONPATH:<path to SSD-pytorch>:<path to aimet-model-zoo>
  ```

### Dataset
COCO 2017 val dataset can be downloaded from here:
- https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset?select=coco2017

---

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python3  aimet_zoo_torch/ssd_res50/evaluators/ssd_res50_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to the downloaded COCO dataset> \
                --use-cuda
```

Available model configurations are:
- ssd_res50_w8a8


---

## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF was used as quantization scheme
- Cross-layer-Equalization have been applied on optimized checkpoint
