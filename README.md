![Qualcomm Innovation Center, Inc.](images/logo-quic-on@h68.png)

# Model Zoo for AI Model Efficiency Toolkit

We provide a collection of popular neural network models and compare their floating point (FP32) and quantized (INT8 weights/activations or INT8 weights/INT16 activations) performance. Results demonstrate that quantized models can provide good accuracy, comparable to floating point (FP32) models. Together with results, we also provide recipes for users to quantize floating-point models using the [AI Model Efficiency ToolKit (AIMET)](https://github.com/quic/aimet).  


## Table of Contents
- [Introduction](#introduction)
- [Tensorflow Models](#tensorflow-models)
  - [Model Zoo](#model-zoo)
  - [Detailed Results](#detailed-results)
- [PyTorch Models](#pytorch-models)
  - [Model Zoo](#pytorch-model-zoo)
  - [Detailed Results](#pytorch-detailed-results)
- [Examples](#examples)
- [Team](#team)
- [License](#license)

## Introduction
Quantized inference is significantly faster than floating-point inference, and enables models to run in a power-efficient manner on mobile and edge devices. We use AIMET, a library that includes state-of-the-art techniques for quantization, to quantize various models available in [TensorFlow](https://tensorflow.org) and [PyTorch](https://pytorch.org) frameworks. The list of models is provided in the sections below.

An original FP32 source model is quantized either using post-training quantization (PTQ) or Quantization-Aware-Training (QAT) technique available in AIMET. Example scripts for evaluation are provided for each model. When PTQ is needed, the evaluation script performs PTQ before evaluation. Wherever QAT is used, the fine-tuned model checkpoint is also provided.

## Tensorflow Models

### Model Zoo
<table style="width:50%">
  <tr>
    <th>Network</th>
    <th>Model Source <sup>[1]</sup></th>
    <th>Floating Pt (FP32) Model <sup>[2]</sup></th>
    <th>Quantized Model <sup>[3]</sup></th>
    <th>Results <sup>[4]</sup></th>
    <th>Documentation</th>
  </tr>
  <tr>
    <td>ResNet-50 (v1)</td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/slim">GitHub Repo</a></td>
    <td><a href="http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz">Pretrained Model</a></td>
    <td><a href="zoo_tensorflow/Docs/ResNet50.md">See Documentation</a></td>
    <td>(ImageNet) Top-1 Accuracy <br>FP32: 75.21% <br> INT8: 74.96%</td>
    <td><a href="zoo_tensorflow/Docs/ResNet50.md">ResNet50.md</a></td>
  </tr>
  <tr>
    <td>MobileNet-v2-1.4</td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/slim">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz">Pretrained Model</a></td>
    <td><a href="/../../releases/download/mobilenet-v2-1.4/mobilenetv2-1.4.tar.gz">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy <br> FP32: 75%<br> INT8: 74.21%</td>
    <td><a href="zoo_tensorflow/Docs/MobileNetV2.md">MobileNetV2.md</a></td>
  </tr>
  <tr>
    <td>EfficientNet Lite</td>
    <td><a href="https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz">Pretrained Model</a </td>
    <td><a href="/../../releases/download/efficientnet-lite0/efficientnet-lite0.tar.gz">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy <br> FP32: 74.93% <br> INT8: 74.99%</td>
    <td><a href="zoo_tensorflow/Docs/EfficientNetLite.md">EfficientNetLite.md</a></td>
  </tr>
  <tr>
    <td>SSD MobileNet-v2</td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/object_detection">GitHub Repo</a></td>
    <td><a href="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz">Pretrained Model</a></td>
    <td><a href="zoo_tensorflow/examples/ssd_mobilenet_v2_quanteval.py">See Example</a></td>
    <td>(COCO) Mean Avg. Precision (mAP) <br> FP32: 0.2469<br> INT8: 0.2456</td>
    <td><a href="zoo_tensorflow/Docs/SSDMobileNetV2.md">SSDMobileNetV2.md</a></td>
  </tr>
  <tr>
    <td>RetinaNet</td>
    <td><a href="https://github.com/fizyr/keras-retinanet">GitHub Repo</a></td>
    <td><a href="https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5">Pretrained Model</a></td>
    <td><a href="zoo_tensorflow/examples/retinanet_quanteval.py">See Example</a></td>
    <td> (COCO) mAP <br> FP32: 0.35 <br> INT8: 0.349 <br><a href="#retinanet"> Detailed Results</a></td>
    <td><a href="zoo_tensorflow/Docs/RetinaNet.md">RetinaNet.md</a></td>
  </tr>
<tr>
    <td>Pose Estimation</td>
    <td><a href="https://arxiv.org/abs/1611.08050">Based on Ref.</a></td>
    <td><a href="https://arxiv.org/abs/1611.08050">Based on Ref.</a></td>
    <td><a href="/../../releases/download/pose_estimation/pose_estimation_tensorflow.tar.gz">Quantized Model</a></td>
    <td>(COCO) mAP <br>FP32: 0.383 <br> INT8: 0.379, <br> Mean Avg.Recall (mAR) <br> FP32: 0.452<br> INT8: 0.446</td>
    <td><a href="zoo_tensorflow/Docs/PoseEstimation.md">PoseEstimation.md</a></td>
  </tr>
  <tr>
    <td>SRGAN</td>
    <td><a href="https://github.com/krasserm/super-resolution">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1u9ituA3ScttN9Vi-UkALmpO0dWQLm8Rv/view">Pretrained Model</a></td>
    <td><a href="zoo_tensorflow/examples/srgan_quanteval.py">See Example</a></td>
    <td>(BSD100) PSNR/SSIM <br> FP32: 25.45/0.668 <br> INT8: 24.78/0.628<br> INT8W/INT16Act.: 25.41/0.666 <br> <a href="#srgan"> Detailed Results</a></td>
    <td><a href="zoo_tensorflow/Docs/SRGAN.md">SRGAN.md</a></td>
  </tr>

</table>

*<sup>[1]</sup>* Original FP32 model source  
*<sup>[2]</sup>* FP32 model checkpoint  
*<sup>[3]</sup>* Quantized Model: For models quantized with post-training technique, refers to FP32 model which can then be quantized using AIMET. For models optimized with QAT, refers to model checkpoint with fine-tuned weights. 8-bit weights and activations are typically used. For some models, 8-bit weights and 16-bit activations (INT8W/INT16Act.) are used to further improve performance of post-training quantization.  
*<sup>[4]</sup>* Results comparing float and quantized performance  
*<sup>[5]</sup>* Script for quantized evaluation using the model referenced in “Quantized Model” column

### Detailed Results
#### RetinaNet
(COCO dataset)
<table style= " width:50%">
   <tr>
    <th>Average Precision/Recall </th>
    <th> @[ IoU | area | maxDets] </th>
    <th>FP32 </th>
     <th>INT8 </th>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.50:0.95 | all | 100 ] </td>
    <td>0.350 </td>
    <td>0.349</td>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.50 | all | 100 ] </td>
    <td>0.537 </td>
    <td>0.536</td>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.75 | all | 100 ] </td>
    <td>0.374 </td>
    <td> 0.372</td>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.50:0.95 | small | 100 ] </td>
    <td>0.191 </td>
    <td>0.187</td>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.50:0.95 | medium | 100 ] </td>
    <td> 0.383 </td>
    <td>0.381</td>
  </tr>
  <tr>
    <td>Average Precision</td>
    <td> @[ 0.50:0.95 | large | 100 ] </td>
    <td>0.472 </td>
    <td>0.472</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[ 0.50:0.95 | all | 1 ] </td>
    <td>0.306 </td>
    <td>0.305</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[0.50:0.95 | all | 10 ] </td>
    <td>0.491 </td>
    <td>0.490</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[ 0.50:0.95 | all |100 ] </td>
    <td>0.533 </td>
    <td>0.532</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[ 0.50:0.95 | small | 100 ] </td>
    <td>0.345</td>
    <td>0.341</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[ 0.50:0.95 | medium | 100 ] </td>
    <td>0.577</td>
    <td>0.577</td>
  </tr>
  <tr>
    <td> Average Recall</td>
    <td> @[ 0.50:0.95 | large | 100 ] </td>
    <td>0.681</td>
    <td>0.679</td>
  </tr>
</table>

#### SRGAN
 <table style= " width:50%">
   <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>PSNR</th>
    <th>SSIM</th>
  </tr>
  <tr>
    <td>FP32</td>
    <td>Set5/Set14/BSD100</td>
    <td>29.17/26.17/25.45</td>
    <td>0.853/0.719/0.668</td>
  </tr>
  <tr>
    <td>INT8/ACT8</td>
    <td>Set5/Set14/BSD100</td>
    <td>28.31/25.55/24.78</td>
    <td>0.821/0.684/0.628</td>
  </tr>
  <tr>
    <td>INT8/ACT16</td>
    <td>Set5/Set14/BSD100</td>
    <td>29.12/26.15/25.41</td>
    <td>0.851/0.719/0.666</td>
  </tr>
</table>


## PyTorch Models
### Model Zoo  <a name="pytorch-model-zoo"></a>
<table style="width:50%">
<tr>
    <th>Network</th>
    <th>Model Source <sup>[1]</sup></th>
    <th>Floating Pt (FP32) Model <sup>[2]</sup></th>
    <th>Quantized Model <sup>[3]</sup></th>
    <th>Results <sup>[4]</sup></th>
    <th>Documentation</th>
  </tr>
  <tr>
    <td>MobileNetV2</td>
    <td><a href="https://github.com/tonylins/pytorch-mobilenet-v2">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR/view">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/eval_mobilenetv2.py">See Example</a></td>
    <td>(ImageNet) Top-1 Accuracy <br>FP32: 71.67%<br> INT8: 71.14%</td>
    <td><a href="zoo_torch/Docs/MobilenetV2.md">MobileNetV2.md</a></td>
  </tr>
  <tr>
    <td>EfficientNet-lite0</td>
    <td><a href="https://github.com/rwightman/gen-efficientnet-pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/eval_efficientnetlite0.py">See Example</a></td>
    <td>(ImageNet) Top-1 Accuracy <br> FP32: 75.42%<br> INT8: 74.49%</td>
    <td><a href="zoo_torch/Docs/EfficientNet-lite0.md">EfficientNet-lite0.md</a></td>
  </tr>
  <tr>
    <td>DeepLabV3+</td>
    <td><a href="https://github.com/jfzhang95/pytorch-deeplab-xception">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/eval_deeplabv3.py">See Example</a></td>
    <td>(PascalVOC) mIOU <br>FP32: 72.32%<br> INT8: 72.08%</a></td>
    <td><a href="zoo_torch/Docs/DeepLabV3.md">DeepLabV3.md</a></td>
  </tr>
  <tr>
    <td>MobileNetV2-SSD-Lite</td>
    <td><a href="https://github.com/qfgaohao/pytorch-ssd">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/eval_ssd.py">See Example</a></td>
    <td>(PascalVOC) mAP<br> FP32: 68.7%<br> INT8: 68.6%</td>
    <td><a href="zoo_torch/Docs/MobileNetV2-SSD-lite.md">MobileNetV2-SSD-lite.md</a></td>
  </tr>
  <tr>
    <td>Pose Estimation</td>
    <td><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td><a href="/../../releases/download/pose_estimation_pytorch/pose_estimation_pytorch.gz">Quantized Model</a></td>
    <td>(COCO) mAP<br>FP32: 0.364<br>INT8: 0.359<br> mAR <br> FP32: 0.436<br> INT8: 0.432</td>
    <td><a href="zoo_torch/Docs/PoseEstimation.md">PoseEstimation.md</a></td>
  </tr>
  <tr> 
    <td>SRGAN</td>
    <td><a href="https://github.com/andreas128/mmsr">GitHub Repo</a></td>
    <td><a href="/../../releases/download/srgan_mmsr_model/srgan_mmsr_MSRGANx4.gz">Pretrained Model</a> (older version from <a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan">here</a>)</td>    
    <td>N/A</td>
    <td>(BSD100) PSNR/SSIM <br> FP32: 25.51/0.653<br> INT8: 25.5/0.648<br><a href="#srgan-pytorch"> Detailed Results</a></td>
    <td><a href="zoo_torch/Docs/SRGAN.md">SRGAN.md</a></td>
  </tr>
  <tr>
    <td>DeepSpeech2</td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/deepspeech2_quanteval.py">See Example</a></td>
    <td>(Librispeech Test Clean) WER <br> FP32<br> 9.92%<br> INT8: 10.22%</td>
    <td><a href="zoo_torch/Docs/DeepSpeech2.md">DeepSpeech2.md</a></td>
  </tr>
</table>

*<sup>[1]</sup>* Original FP32 model source  
*<sup>[2]</sup>* FP32 model checkpoint  
*<sup>[3]</sup>* Quantized Model: For models quantized with post-training technique, refers to FP32 model which can then be quantized using AIMET. For models optimized with QAT, refers to model checkpoint with fine-tuned weights. 8-bit weights and activations are typically used. For some models, 8-bit weights and 16-bit weights are used to further improve performance of post-training quantization.  
*<sup>[4]</sup>* Results comparing float and quantized performance  
*<sup>[5]</sup>* Script for quantized evaluation using the model referenced in “Quantized Model” column

### Detailed Results <a name="pytorch-detailed-results"></a>

#### SRGAN Pytorch <a name="srgan-pytorch"></a>
<table style= " width:50%">
   <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>PSNR</th>
    <th>SSIM</th>
  </tr>
  <tr>
    <td>FP32</td>
    <td>Set5/Set14/BSD100</td>
    <td>29.93/N/A/25.51</td>
    <td>0.851/N/A/0.653</td>
  </tr>
  <tr>
    <td>INT8</td>
    <td>Set5/Set14/BSD100</td>
    <td>29.86/N/A/25.55</td>
    <td>0.845/N/A/0.648</td>
  </tr>
</table>


## Examples

### Install AIMET
Before you can run the example script for a specific model, you need to install the AI Model Efficiency ToolKit (AIMET) software. Please see this [Getting Started](https://github.com/quic/aimet#getting-started) page for an overview. Then install AIMET and its dependencies using these [Installation instructions](https://github.com/quic/aimet/blob/1.13.0/packaging/INSTALL.txt).

> **NOTE:** To obtain the exact version of AIMET software that was used to test this model zoo, please install release [1.13.0](https://github.com/quic/aimet/releases/tag/1.13.0) when following the above instructions.

### Running the scripts
Download the necessary datasets and code required to run the example for the model of interest. The examples run quantized evaluation and if necessary apply AIMET techniques to improve quantized model performance. They generate the final accuracy results noted in the table above. Refer to the Docs for [TensorFlow](zoo_tensorflow/Docs) or [PyTorch](zoo_torch/Docs) folder to access the documentation and procedures for a specific model.

## Team
AIMET Model Zoo is a project maintained by Qualcomm Innovation Center, Inc.

## License
Please see the [LICENSE file](LICENSE.pdf) for details.
