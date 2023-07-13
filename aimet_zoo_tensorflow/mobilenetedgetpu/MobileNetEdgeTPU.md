# MobileNet-EdgeTPU

## Environment Setup 

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.26/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.26.0 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.26.0) i.e. please set `release_tag="1.26.0"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

### Append the repo location to your `PYTHONPATH` by doing the following:
  `export PYTHONPATH=$PYTHONPATH:/<path to parent>/aimet-model-zoo`

## Dataset
- ImageNet can be downloaded from here:
  - http://www.image-net.org/
- For this evaluation, Tf-records of ImageNet validation dataset are required. (See https://github.com/tensorflow/models/tree/master/research/slim#Data for details)
- The Tf-records of ImageNet validation dataset should be organized in the following way
```bash
< path to ImageNet validation dataset Tf-records>
├── validation-00000-of-00128
├── validation-00001-of-00128
├── ...
```
  

---

## Model checkpoint for AIMET optimization
 - Downloading of model checkpoints is handled by evaluation script.
 - Checkpoint used for AIMET quantization can be downloaded from the [Releases](/../../releases) page.

 ---

## Usage
```bash
python aimet_zoo_tensorflow/mobilenetedgetpu/evaluators/mobilenet_edgetpu_quanteval.py 
 --model-config <model configuration to test> \ 
 --dataset-path <path to tfrecord dataset> 
 --batch-size <batch size for inference>
```

* example 
    ```
    python mobilenet_edgetpu_quanteval.py  --dataset-path <ILSVRC2012_PyTorch_path> --model-config mobilenetedgetpu_w8a8 --batch-size 64
    ```

---

## Quantization configuration 
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:

+ Weight quantization: 8 bits, per-channel symmetric quantization
+ Bias parameters are not quantized
+ Activation quantization: 8 bits, asymmetric quantization
+ TF_enhanced was used for weight quantization scheme
+ TF_enhanced was used for activation quantization scheme
+ Batch_norm_fold was used for weight
