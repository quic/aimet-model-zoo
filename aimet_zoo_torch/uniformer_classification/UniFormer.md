# Uniformer for Image Classification

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.26/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.26.

### Environment Setup 
Append the repo location to your `PYTHONPATH` with the following:  
  ```bash
  export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo>
  ```

### Dataset 
The ImageNet 2012 Challenge (ILSVRC2012) dataset can be obtained from :
  - https://www.image-net.org/download.php


---

## Usage
```bash
python3  aimet_zoo_torch/uniformer_classification/evaluators/uniformer_classification_quanteval.py \
                --model-config <configuration to be tested> \
                --dataset-path <path to the downloaded Imagenet dataset, should contain the 'train' and 'val' subdirectories> \
                --batch-size <batch size as an integer value, defaults to 32> \
```

Available model configurations are:
- uniformer_classification_w8a8

---


## Model checkpoint and configuration

Individual artifacts can be obtained from:
 - https://github.com/quic/aimet-model-zoo/releases/tag/torch_uniformer_classification

---

## Quantization Configuration
The following configuration has been used for both W4A8 and W8A8 variants:
- Weight quantization: 8 bits, per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- training_range_learning_with_tf_init was used as quantization scheme
- Batch Norm Fold has been applied on optimized checkpoint
- Quantization Aware Training has been performed on the optimized checkpoint
