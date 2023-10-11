![Qualcomm Innovation Center, Inc.](images/logo-quic-on@h68.png)

# Model Zoo for AI Model Efficiency Toolkit
We provide a collection of popular neural network models and compare their floating point and quantized performance. Results demonstrate that quantized models can provide good accuracy, comparable to floating point models. Together with results, we also provide scripts and artifacts for users to quantize floating-point models using the [AI Model Efficiency ToolKit (AIMET)](https://github.com/quic/aimet).

## Table of Contents
- [Introduction](#introduction)
- [PyTorch Models](#pytorch-models)
- [Tensorflow Models](#tensorflow-models)
- [Installation and Usage](#installation-and-usage)
- [Team](#team)
- [License](#license)

## Introduction
Quantized inference is significantly faster than floating-point inference, and enables models to run in a power-efficient manner on mobile and edge devices. We use AIMET, a library that includes state-of-the-art techniques for quantization, to quantize various models available in [PyTorch](https://pytorch.org) and [TensorFlow](https://tensorflow.org) frameworks.

An original FP32 source model is quantized either using post-training quantization (PTQ) or Quantization-Aware-Training (QAT) technique available in AIMET. Example scripts for evaluation are provided for each model. When PTQ is needed, the evaluation script performs PTQ before evaluation. Wherever QAT is used, the fine-tuned model checkpoint is also provided.

## PyTorch Models
<table style="width:50%;text-align: center">
  <tr>
    <th>Task</th>
    <th>Network<sup>[1]</sup></th>
    <th>Model Source<sup>[2]</sup></th>
    <th>Floating Pt (FP32) Model <sup>[3]</sup></th>
    <th>Quantized Model <sup>[4]</sup></th>
    <th colspan="4">Results <sup>[5]</sup></th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th>Metric</th>
    <th>FP32</th>
    <th>W8A8<sup>[6]</sup></th>
    <th>W4A8<sup>[7]</sup></th>
  </tr>
  <tr>
    <td rowspan="12">Image Classification</td>
    <td><a href="aimet_zoo_torch/mobilenetv2/MobilenetV2.md">MobileNetV2</a></td>
    <td><a href="https://github.com/tonylins/pytorch-mobilenet-v2">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR/view">Pretrained Model</a></td>
    <td><a href="/../../releases/download/mobilenetv2-pytorch/mv2qat_modeldef.tar.gz">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>71.67%</td>
    <td>71.14%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/resnet/ResNet.md">Resnet18</a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="/../../releases/tag/torchvision_classification_INT4%2F8">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>69.75%</td>
    <td>69.54%</td>
    <td>69.1%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/resnet/ResNet.md">Resnet50</a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="/../../releases/tag/torchvision_classification_INT4%2F8">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>76.14%</td>
    <td>75.81%</td> 
    <td>75.63%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/resnet/ResNet.md">Resnet101</a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="/../../releases/tag/torch_resnet101_w8a8">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>77.34%</td>
    <td>77.13%</td> 
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/regnet/RegNet.md">Regnet_x_3_2gf</a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="/../../releases/tag/torchvision_classification_INT4%2F8">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>78.36%</td>
    <td>78.10%</td>
    <td>77.70%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/resnext/ResNeXt.md">ResNeXt101</a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="https://pytorch.org/vision/0.11/models.html#classification">Pytorch Torchvision </a></td>
    <td><a href="/../../releases/tag/torch_resnext101">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>79.23%</td>
    <td>78.76%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/hrnet_image_classification/HRNet_Image_Classification.md">HRNet_W32</a></td>
    <td><a href="https://github.com/HRNet/HRNet-Image-Classification">GitHub Repo</a></td>
    <td><a href="https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ">Pretrained Model </a></td>
    <td><a href="/../../releases/tag/torch_hrnetw32_w8a8">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>78.50%</td>
    <td>78.20%</td> 
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/efficientnetlite0/EfficientNet-lite0.md">EfficientNet-lite0</a></td>
    <td><a href="https://github.com/rwightman/gen-efficientnet-pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_effnet_lite0_w8a8_pc">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>75.40%</td>
    <td>75.36%</td>
    <td>74.46%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/vit/ViT.md">ViT</a></td>
    <td><a href="https://huggingface.co/docs/transformers/model_doc/vit">Repo</a></td>
    <td><a href="/../../releases/tag/torch_vit">Prepared Models </a></td>
    <td><a href="/../../releases/tag/torch_vit">Quantized Models</a></td>    
    <td>(ImageNet dataset) Accuracy</td>
    <td>81.32</td>
    <td>81.57</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/mobilevit/MobileViT.md">MobileViT</a></td>
    <td><a href="https://huggingface.co/docs/transformers/model_doc/mobilevit">Repo</a></td>
    <td><a href="/../../releases/tag/torch_mobilevit">Prepared Models </a></td>
    <td ><a href="/../../releases/tag/torch_mobilevit">Quantized Models</a></td>
    <td>(ImageNet dataset) Accuracy</td>
    <td>78.46</td>
    <td>77.59</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/gpunet0/GPUNet0.md">GPUNet</a></td>
    <td><a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/GPUNet">Repo</a></td>
    <td><a href="/../../releases/tag/torch_gpunet0_w8a8">Prepared Models </a></td>
    <td><a href="/../../releases/tag/torch_gpunet0_w8a8">Quantized Models</a></td>
    <td>(ImageNet dataset) Accuracy</td>
    <td>78.86</td>
    <td>78.42</td>
    <td>TBD</td>
  </tr>
    <tr>
    <td><a href="aimet_zoo_torch/uniformer_classification/Uniformer.md">Uniformer</a></td>
    <td><a href="https://github.com/Sense-X/UniFormer">Repo</a></td>
    <td><a href="/../../releases/tag/torch_uniformer_classification">Prepared Models </a></td>
    <td><a href="/../../releases/tag/torch_uniformer_classification">Quantized Models</a></td>
    <td>(ImageNet dataset) Accuracy</td>
    <td>82.9</td>
    <td>81.9</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="3">Object Detection</td>
    <td><a href="aimet_zoo_torch/ssd_mobilenetv2/MobileNetV2-SSD-lite.md">MobileNetV2-SSD-Lite</a></td>
    <td><a href="https://github.com/qfgaohao/pytorch-ssd">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/phase_2_january_artifacts">Quantized Model</a></td>
    <td>(PascalVOC) mAP</td>
    <td>68.7%</td>
    <td>68.6%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/ssd_res50/SSD_Res50.md">SSD_Res50</a></td>
    <td><a href="https://github.com/qfgaohao/pytorch-ssd">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_DYYDJUfwLIvGBDnM3hMFNgkVRZW6MgX">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_ssd_res50">Quantized Model</a></td>
    <td>(COCO2017val) mAP</td>
    <td>0.250</td>
    <td>0.248</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/yolox/YOLOX.md">YOLOX</a></td>
    <td><a href="https://github.com/Megvii-BaseDetection/YOLOX">Github Repo</a></td>
    <td><a href="/../../releases/tag/torch_yolox_int8">Pretrained Models (2 in total)</a></td>
    <td><a href="/../../releases/tag/torch_yolox_int8">Quantized Model</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/yolox/YOLOX.md#results">mAP Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="4">Pose Estimation</td>
    <td rowspan="2"><a href="aimet_zoo_torch/poseestimation/PoseEstimation.md">Pose Estimation</a></td>
    <td rowspan="2"><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td rowspan="2"><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td rowspan="2"><a href="/../../releases/download/torch_pose_estimation/pose_estimation_pytorch_weights.tgz">Quantized Model</a></td>
    <td>(COCO) mAP</td>
    <td>0.364</td>
    <td>0.359</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(COCO) mAR</td>
    <td>0.436</td>
    <td>0.432</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="aimet_zoo_torch/hrnet_posenet/Hrnet-posenet.md">HRNET-Posenet</a></td>
    <td rowspan="2"><a href="https://github.com/leoxiaobin/deep-high-resolution-net.pytorch">Based on Ref.</a></td>
    <td rowspan="2"><a href="/../../releases/tag/phase_2_march_artifacts">FP32 Model</a></td>
    <td rowspan="2"><a href="/../../releases/tag/phase_2_march_artifacts">Quantized Model</a></td>
    <td>(COCO) mAP</td>
    <td>0.765</td>
    <td>0.763</td>
    <td>0.762</td>
  </tr>
  <tr>
    <td>(COCO) mAR</td>
    <td>0.793</td>
    <td>0.792</td>
    <td>0.791</td>
  </tr>
  <tr>
    <td rowspan="5">Super Resolution</td>
    <td><a href="aimet_zoo_torch/srgan/SRGAN.md">SRGAN</a></td>
    <td><a href="https://github.com/andreas128/mmsr">GitHub Repo</a></td>
    <td><a href="/../../releases/download/torch_srgan_mmsr_model/srgan_mmsr_MSRGANx4.gz">Pretrained Model</a> (older version from <a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan">here</a>)</td>    
    <td><a href="aimet_zoo_torch/srgan/evaluators/srgan_quanteval.py">See Example</a></td>
    <td>(BSD100) PSNR / SSIM <a href="aimet_zoo_torch/Docs/SRGAN.md#results"> Detailed Results</a></td>
    <td>25.51 / 0.653</td>
    <td>25.5 / 0.648</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/abpn/ABPN.md">Anchor-based Plain Net (ABPN)</a></td>
    <td><a href="https://arxiv.org/abs/2105.09750">Based on Ref.</a></td>
    <td><a href="/../../releases/tag/torch_abpn_checkpoint">See Tarballs</a></td>
    <td><a href="aimet_zoo_torch/abpn/evaluators/abpn_quanteval.py">See Example</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/abpn/ABPN.md#results"> Average PSNR Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/xlsr/XLSR.md">Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution (XLSR)</a></td>
    <td><a href="https://arxiv.org/abs/2105.10288">Based on Ref.</a></td>
    <td><a href="/../../releases/tag/torch_xlsr_checkpoint">See Tarballs</a></td>
    <td><a href="aimet_zoo_torch/xlsr/evaluators/xlsr_quanteval.py">See Example</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/xlsr/XLSR.md#results"> Average PSNR Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/sesr/SESR.md">Super-Efficient Super Resolution (SESR)</a></td>
    <td><a href="https://arxiv.org/abs/2103.09404">Based on Ref.</a></td>
    <td><a href="/../../releases/tag/torch_sesr_checkpoint">See Tarballs</a></td>
    <td><a href="aimet_zoo_torch/sesr/evaluators/sesr_quanteval.py">See Example</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/sesr/SESR.md#results"> Average PSNR Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/quicksrnet/QuickSRNet.md">QuickSRNet</a></td>
    <td><a> - </a></td>
    <td><a href="/../../releases/tag/phase_2_january_artifacts">See Tarballs</a></td>
    <td><a href="aimet_zoo_torch/quicksrnet/evaluators/quicksrnet_quanteval.py">See Example</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/quicksrnet/QuickSRNet.md#results"> Average PSNR Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="8">Semantic Segmentation</td>
    <td><a href="aimet_zoo_torch/deeplabv3/DeepLabV3.md">DeepLabV3+</a></td>
    <td><a href="https://github.com/jfzhang95/pytorch-deeplab-xception">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_dlv3_w8a8_pc">Quantized Model</a></td>
    <td>(PascalVOC) mIOU</td>
    <td>72.91%</td>
    <td>72.44%</td>
    <td>72.18%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/hrnet_semantic_segmentation/HRNet_Semantic_Segmentation.md">HRNet-W48</a></td>
    <td><a href="https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1">GitHub Repo</a></td>
    <td> Original model weight not available </a></td>
    <td><a href="/../../releases/tag/phase_2_february_artifacts">Quantized Model</a></td>
    <td>(Cityscapes) mIOU</td>
    <td>81.04%</td>
    <td>80.65%</td>
    <td>80.07%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/inverseform/InverseForm.md">InverseForm (HRNet-16-Slim-IF)</a></td>
    <td><a href="https://github.com/Qualcomm-AI-research/InverseForm">GitHub Repo</a></td>
    <td><a href="https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hr16s_4k_slim.pth">Pretrained Model</a></td>
    <td><a href="aimet_zoo_torch/inverseform/evaluators/inverseform_quanteval.py">See Example</a></td>
    <td>(Cityscapes) mIOU</td>
    <td>77.81%</td>
    <td>77.17%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/inverseform/InverseForm.md">InverseForm (OCRNet-48)</a></td>
    <td><a href="https://github.com/Qualcomm-AI-research/InverseForm">GitHub Repo</a></td>
    <td><a href="https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hrnet48_OCR_IF_checkpoint.pth">Pretrained Model</a></td>
    <td><a href="aimet_zoo_torch/inverseform/evaluators/inverseform_quanteval.py">See Example</a></td>
    <td>(Cityscapes) mIOU</td>
    <td>86.31%</td>
    <td>86.21%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/ffnet/FFNet.md">FFNets</a></td>
    <td><a href="https://github.com/Qualcomm-AI-research/FFNet"> Github Repo</a></td>
    <td><a href="/../../releases/tag/torch_segmentation_ffnet">Prepared Models (5 in total)</a></td>
    <td><a href="aimet_zoo_torch/ffnet/evaluators/ffnet_quanteval.py">See Example</a></td>
    <td colspan="3"><a href="aimet_zoo_torch/ffnet/FFNet.md#results">mIoU Results</a></td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/rangenet/rangeNet.md">RangeNet++</a></td>
    <td><a href="https://github.com/PRBonn/lidar-bonnetal">GitHub Repo</a></td>
    <td><a href="http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet21.tar.gz">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_rangenet_plus_w8a8">Quantized Model</a></td>
    <td>(Semantic kitti) mIOU</td>
    <td>47.2%</td>
    <td>47.1%</td>
    <td>46.8%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/salsanext/SalsaNext.md">SalsaNext</a></td>
    <td><a href="https://github.com/TiagoCortinhal/SalsaNext">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/10fxIwPK10UVVB9jsgXDZSDwj4vy9MyTl/view">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_salsanext">Quantized Model</a></td>
    <td>(Semantic kitti) mIOU</td>
    <td>55.8%</td>
    <td>54.9%</td>
    <td>55.1%</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/segnet/SegNet.md">SegNet</a></td>
    <td><a href="https://github.com/alexgkendall/SegNet-Tutorial">GitHub Repo</a></td>
    <td><a href="/../../releases/tag/torch_segnet">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_segnet">Quantized Model</a></td>
    <td>(CamVid dataset) mIOU</td>
    <td>50.48%</td>
    <td>50.59%</td>
    <td>50.58%</td>
  </tr>
  <tr>
    <td rowspan="1">Video Understanding</td>
    <td><a href="aimet_zoo_torch/mmaction2/mmaction2.md">mmaction2 BMN</a></td>
    <td><a href="https://github.com/open-mmlab/mmaction2">GitHub Repo</a></td>
    <td><a href="https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature_20220908-79f92857.pth">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/torch_mmaction2">Quantized Model</a></td>
    <td>(ActivityNet) auc</td>
    <td>67.25</td>
    <td>67.05</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>Speech Recognition</td>
    <td><a href="aimet_zoo_torch/deepspeech2/DeepSpeech2.md">DeepSpeech2</a></td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth">Pretrained Model</a></td>
    <td><a href="aimet_zoo_torch/deepspeech2/evaluators/deepspeech2_quanteval.py">See Example</a></td>
    <td>(Librispeech Test Clean) WER</td>
    <td>9.92%</td>
    <td>10.22%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="15">NLP / NLU</td>
    <td rowspan="3"><a href="aimet_zoo_torch/bert/Bert.md">Bert</a></td>
    <td rowspan="3"><a href="https://huggingface.co/docs/transformers/model_doc/bert">Repo</a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_bert">Prepared Models </a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_bert">Quantized Models</a></td>
    <td>(GLUE dataset) GLUE score</td>
    <td>83.11</td>
    <td>82.44</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(SQuAD dataset) F1 score</td>
    <td>88.48</td>
    <td>87.47</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td colspan="3"><a href="aimet_zoo_torch/bert/Bert.md#results"> Detailed Results</a></td>
  </tr>
  <tr>
    <td rowspan="3"><a href="aimet_zoo_torch/mobilebert/MobileBert.md">MobileBert</a></td>
    <td rowspan="3"><a href="https://huggingface.co/docs/transformers/model_doc/mobilebert">Repo</a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_mobilebert">Prepared Models </a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_mobilebert">Quantized Models</a></td>
    <td>(GLUE dataset) GLUE score</td>
    <td>81.24</td>
    <td>81.17</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(SQuAD dataset) F1 score</td>
    <td>89.45</td>
    <td>88.66</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td colspan="3"><a href="aimet_zoo_torch/mobilebert/MobileBert.md#results"> Detailed Results</a></td>
  </tr>
  <tr>
    <td rowspan="3"><a href="aimet_zoo_torch/minilm/MiniLM.md">MiniLM</a></td>
    <td rowspan="3"><a href="https://huggingface.co/microsoft/MiniLM-L12-H384-uncased?text=I+like+you.+I+love+you">Repo</a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_minilm">Prepared Models </a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_minilm">Quantized Models</a></td>
    <td>(GLUE dataset) GLUE score</td>
    <td>82.23</td>
    <td>82.63</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(SQuAD dataset) F1 score</td>
    <td>90.47</td>
    <td>89.70</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td colspan="3"><a href="aimet_zoo_torch/minilm/MiniLM.md#results"> Detailed Results</a></td>
  </tr>
  <tr>
    <td rowspan="2"><a href="aimet_zoo_torch/roberta/Roberta.md">Roberta</a></td>
    <td rowspan="2"><a href="https://huggingface.co/docs/transformers/model_doc/roberta">Repo</a></td>
    <td rowspan="2"><a href="/../../releases/tag/torch_roberta">Prepared Models </a></td>
    <td rowspan="2"><a href="/../../releases/tag/torch_roberta">Quantized Models</a></td>
    <td>(GLUE dataset) GLUE score</td>
    <td>85.11</td>
    <td>84.26</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td colspan="3"><a href="aimet_zoo_torch/roberta/Roberta.md#results"> Detailed Results</a></td>
  </tr>
  <tr>
    <td rowspan="3"><a href="aimet_zoo_torch/distilbert/DistilBert.md">DistilBert</a></td>
    <td rowspan="3"><a href="https://huggingface.co/docs/transformers/model_doc/distilbert">Repo</a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_distilbert">Prepared Models </a></td>
    <td rowspan="3"><a href="/../../releases/tag/torch_distilbert">Quantized Models</a></td>
    <td>(GLUE dataset) GLUE score</td>
    <td>80.71</td>
    <td>80.26</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(SQuAD dataset) F1 score</td>
    <td>85.42</td>
    <td>85.18</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td colspan="3"><a href="aimet_zoo_torch/distilbert/DistilBert.md#results"> Detailed Results</a></td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_torch/gpt2/GPT2.md">GPT2</a></td>
    <td><a href="https://huggingface.co/gpt2">Repo</a></td>
    <td><a href="/../../releases/tag/torch_gpt2">Prepared Models</a></td>
    <td><a href="/../../releases/tag/torch_gpt2">Quantized Models</a></td>
    <td>Perplexity</td>
    <td>27.67</td>
    <td>28.11</td>
    <td>TBD</td>
  </tr>

</table>

*<sup>[1]</sup>* <sub>Model usage documentation</sub>  
*<sup>[2]</sup>* <sub>Original FP32 model source</sub>  
*<sup>[3]</sup>* <sub>FP32 model checkpoint</sub>  
*<sup>[4]</sup>* <sub>Quantized Model: For models quantized with post-training technique, refers to FP32 model which can then be quantized using AIMET. For models optimized with QAT, refers to model checkpoint with fine-tuned weights. 8-bit weights and activations are typically used. For some models, 8-bit weights and 16-bit weights are used to further improve performance of post-training quantization.</sub>  
*<sup>[5]</sup>* <sub>Results comparing float and quantized performance</sub>  
*<sup>[6]</sup>* <sub>W8A8 indicates 8-bit weights, 8-bit activations</sub>  
*<sup>[7]</sup>* <sub>W4A8 indicates 4-bit weights, 8-bit activations (Some models include a mix of W4A8 and W8A8 layers).</sub>  
<sub>TBD indicates that support is NOT yet available</sub>

## Tensorflow Models
<table style="width:50%;text-align: center">
  <tr>
    <th>Task</th>
    <th>Network <sup>[1]</sup></th>
    <th>Model Source <sup>[2]</sup></th>
    <th>Floating Pt (FP32) Model <sup>[3]</sup></th>
    <th>Quantized Model <sup>[4]</sup></th>
    <th>TensorFlow Version</th>
    <th colspan="4">Results <sup>[5]</sup></th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th>Metric</th>
    <th>FP32</th>
    <th>W8A8<sup>[6]</sup></th>
    <th>W4A8<sup>[7]</sup></th>
  </tr>
  <tr>
    <td rowspan="5">Image Classification</td>
    <td><a href="aimet_zoo_tensorflow/resnet50/ResNet50.md">ResNet-50 (v1)</a></td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/slim">GitHub Repo</a></td>
    <td><a href="http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/resnet50/ResNet50.md">See Documentation</a></td>
    <td>1.15</td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>75.21%</td>
    <td>74.96%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/resnet50_tf2/ResNet50_TF2.md">ResNet-50-tf2</a></td>
    <td><a href="https://keras.io/api/applications/resnet/#resnet50-function">GitHub Repo</a></td>
    <td><a href="https://keras.io/api/applications/resnet/#resnet50-function">Pretrained Model</a></td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/archive/refs/tags/tensorflow2-resnet50.tar.gz">Quantized Model</a></td>
    <td>2.4</td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>74.9%</td>
    <td>74.8%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/mobilenet_v2/MobileNetV2.md">MobileNet-v2-1.4</a></td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/slim">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz">Pretrained Model</a></td>
    <td><a href="/../../releases/tag/phase2_january_artifacts">Quantized Model</a></td>
    <td>1.15</td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>75%</td>
    <td>74.21%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/mobilenet_v2_tf2/MobileNetV2_TF2.md">MobileNet-v2-tf2</a></td>
    <td><a href="https://keras.io/api/applications/mobilenet/#mobilenetv2-function">GitHub Repo</a></td>
    <td><a href="https://keras.io/api/applications/mobilenet/#mobilenetv2-function">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/mobilenet_v2_tf2/evaluators/mobilenet_v2_tf2_quanteval.py">See Example</a></td>
    <td>2.4</td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>71.6%</td>
    <td>71.0%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/efficientnet/EfficientNetLite.md">EfficientNet Lite</a></td>
    <td><a href="https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz">Pretrained Model</a> </td>
    <td><a href="/../../releases/download/efficientnet-lite0/efficientnet-lite0.tar.gz">Quantized Model</a></td>
    <td>2.4</td>
    <td>(ImageNet) Top-1 Accuracy</td>
    <td>74.93%</td>
    <td>74.99%</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="3">Object Detection</td>
    <td><a href="aimet_zoo_tensorflow/ssd_mobilenet_v2/SSDMobileNetV2.md">SSD MobileNet-v2</a></td>
    <td><a href="https://github.com/tensorflow/models/tree/master/research/object_detection">GitHub Repo</a></td>
    <td><a href="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/ssd_mobilenet_v2/evaluators/ssd_mobilenet_v2_quanteval.py">See Example</a></td>
    <td>1.15</td>
    <td>(COCO) Mean Avg. Precision (mAP)</td>
    <td>0.2469</td>
    <td>0.2456</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/retinanet/RetinaNet.md">RetinaNet</a></td>
    <td><a href="https://github.com/fizyr/keras-retinanet">GitHub Repo</a></td>
    <td><a href="https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/retinanet/evaluators/retinanet_quanteval.py">See Example</a></td>
    <td>1.15</td>
    <td>(COCO) mAP <a href="aimet_zoo_tensorflow/retinanet/RetinaNet.md#results"> Detailed Results</a></td>
    <td>0.35</td>
    <td>0.349</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="aimet_zoo_tensorflow/mobiledetedgetpu/MobileDetEdgeTPU.md">MobileDet-EdgeTPU</a></td>
    <td><a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md">GitHub Repo</a></td>
    <td><a href="http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/mobiledetedgetpu/evaluators/mobiledet_edgetpu_quanteval.py">See Example</a></td>
    <td>2.4</td>
    <td>(COCO) Mean Avg. Precision (mAP)</td>
    <td>0.281</td>
    <td>0.279</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td rowspan="2">Pose Estimation</td>
    <td rowspan="2"><a href="aimet_zoo_tensorflow/pose_estimation/PoseEstimation.md">Pose Estimation</a></td>
    <td rowspan="2"><a href="https://arxiv.org/abs/1611.08050">Based on Ref.</a></td>
    <td rowspan="2"><a href="https://arxiv.org/abs/1611.08050">Based on Ref.</a></td>
    <td rowspan="2"><a href="/../../releases/download/tf_pose_estimation/pose_estimation_tensorflow.tar.gz">Quantized Model</a></td>
    <td rowspan="2">2.4</td>
    <td>(COCO) mAP</td>
    <td>0.383</td>
    <td>0.379</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>(COCO) (mAR)</td>
    <td>0.452</td>
    <td>0.446</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>Super Resolution</td>
    <td><a href="aimet_zoo_tensorflow/srgan/SRGAN.md">SRGAN</a></td>
    <td><a href="https://github.com/krasserm/super-resolution">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1u9ituA3ScttN9Vi-UkALmpO0dWQLm8Rv/view">Pretrained Model</a></td>
    <td><a href="aimet_zoo_tensorflow/srgan/evaluators/srgan_quanteval.py">See Example</a></td>
    <td>2.4</td>
    <td>(BSD100) PSNR / SSIM <a href="aimet_zoo_tensorflow/srgan/SRGAN.md#results"> Detailed Results</a>
    <td>25.45 / 0.668
    <td>24.78 / 0.628
    <td>25.41 / 0.666 (INT8W / INT16Act.)</td>
  </tr>
  <tr>
    <td rowspan="2">Semantic Segmentation</td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/blob/deeplabv3plus/aimet_zoo_tensorflow/deeplabv3plus_tf2/Deeplabv3plus_TF2.md">DeeplabV3plus_mbnv2</a></td>
    <td><a href="https://github.com/bubbliiiing/deeplabv3-plus-tf2">GitHub Repo</a></td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/releases/tag/tensorflow2-deeplabv3plus_mbnv2">Pretrained Model</a></td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/blob/deeplabv3plus/aimet_zoo_tensorflow/deeplabv3plus_tf2/evaluators/deeplabv3plus_tf2_quanteval.py">See Example</a></td>
    <td>2.4</td>
    <td>(PascalVOC) mIOU</td>
    <td>72.28</td>
    <td>71.71</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/blob/deeplabv3plus/aimet_zoo_tensorflow/deeplabv3plus_tf2/Deeplabv3plus_TF2.md">DeeplabV3plus_xception</a></td>
    <td><a href="https://github.com/bubbliiiing/deeplabv3-plus-tf2">GitHub Repo</a></td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/releases/tag/tensorflow2-deeplabv3plus_xception">Pretrained Model</a></td>
    <td><a href="https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/blob/deeplabv3plus/aimet_zoo_tensorflow/deeplabv3plus_tf2/evaluators/deeplabv3plus_tf2_quanteval.py">See Example</a></td>
    <td>2.4</td>
    <td>(PascalVOC) mIOU</td>
    <td>87.71</td>
    <td>87.21</td>
    <td>TBD</td>
  </tr>
</table>

*<sup>[1]</sup>* <sub>Model usage documentation</sub>  
*<sup>[2]</sup>* <sub>Original FP32 model source</sub>  
*<sup>[3]</sup>* <sub>FP32 model checkpoint</sub>  
*<sup>[4]</sup>* <sub>Quantized Model: For models quantized with post-training technique, refers to FP32 model which can then be quantized using AIMET. For models optimized with QAT, refers to model checkpoint with fine-tuned weights. 8-bit weights and activations are typically used. For some models, 8-bit weights and 16-bit activations (INT8W/INT16Act.) are used to further improve performance of post-training quantization.</sub>  
*<sup>[5]</sup>* <sub>Results comparing float and quantized performance</sub>  
*<sup>[6]</sup>* <sub>W8A8 indicates 8-bit weights, 8-bit activations</sub>  
*<sup>[7]</sup>* <sub>W4A8 indicates 4-bit weights, 8-bit activations (Some models include a mix of W4A8 and W8A8 layers).</sub>  
<sub>TBD indicates that support is NOT yet available</sub>

## Installation and Usage

### Install AIMET
Before you can run the evaluation script for a specific model, you need to install the AI Model Efficiency ToolKit (AIMET) software. Please see this [Getting Started](https://github.com/quic/aimet#getting-started) page for an overview. Then install AIMET and its dependencies using these [Installation instructions](https://quic.github.io/aimet-pages/releases/latest/install).

### Install AIMET model zoo
Follow the instructions on [this page](packaging/README.md) to install the AIMET model zoo python package(s).

### Run model evaluation
The evaluation scripts run floating-point and quantized evaluations that demonstrate improved quantized model performance through the use of AIMET techniques. They generate and display the final accuracy results (as documented in the table above). To access the documentation and procedures for a specific model, refer to the relevant *<model>.md* within the <model> subfolder in [TensorFlow](aimet_zoo_tensorflow) or [PyTorch](aimet_zoo_torch) folders.

## Team
AIMET Model Zoo is a project maintained by Qualcomm Innovation Center, Inc.

## License
Please see the [LICENSE file](LICENSE.pdf) for details.

