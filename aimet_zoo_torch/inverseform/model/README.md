# InverseForm

This repository contains a version of the InverseForm module.

Shubhankar Borse, Ying Wang, Yizhe Zhang, Fatih Porikli, "InverseForm: A Loss Function for Structured Boundary-Aware Segmentation
", CVPR 2021.[[arxiv]](https://arxiv.org/abs/2104.02745)

Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc)

## Reference
If you find our work useful for your research, please cite:
```latex
@inproceedings{borse2021inverseform,
  title={InverseForm: A Loss Function for Structured Boundary-Aware Segmentation},
  author={Borse, Shubhankar and Wang, Ying and Zhang, Yizhe and Porikli, Fatih
},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Method
InverseForm is a novel boundary-aware loss term for semantic segmentation, which efficiently learns the degree of parametric transformations between estimated and target boundaries. 

![! an image](display/inverseform_framework.png)

This plug-in loss term complements the cross-entropy loss in capturing boundary transformations and allows consistent and significant performance improvement on segmentation backbone models without increasing their size and computational complexity.

Here is an example demo from our state-of-the-art model trained on the Cityscapes benchmark.

<img src="display/if_photos_gif.gif " width="425"/> <img src="display/if_labels_gif.gif " width="425"/>

This repository contains the implementation of InverseForm module presented in the paper. It can also run inference on Cityscapes validation set with models trained using the InverseForm framework. The same models can be validated by removing the InverseForm framework such that no additional compute is added during inference. Here are some of the models over which you can run inference with and without the InverseForm block (right-most column of the table below):



| Model           | mIoU (trained w/o InverseForm)  | mIoU (trained w/ InverseForm)   | Checkpoint |
| :-------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: |
| HRNet-18        | 77.0%                           | 77.6% |[hrnet18_IF_checkpoint.pth](https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hrnet18_IF_checkpoint.pth)|
| HRNet-16-Slim   | 76.1%                           | 77.8% |[hr16s_4k_slim.pth](https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hr16s_4k_slim.pth)|
| OCRNet-48       | 86.0%                           | 86.3% | [hrnet48_OCR_IF_checkpoint.pth](https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hrnet48_OCR_IF_checkpoint.pth)|
| OCRNet-48-HMS   | 86.7%                           | 87.0%                           | [hrnet48_OCR_HMS_IF_checkpoint.pth](https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hrnet48_OCR_HMS_IF_checkpoint.pth) |


## Setup environment 

Code has been tested with pytorch 1.3 and NVIDIA Apex. The Dockerfile is available under docker/ folder.

## Cityscapes path   

utils/config.py has the dataset/directory information. Please update CITYSCAPES_DIR as the preferred Cityscapes directory. You can download this dataset from https://www.cityscapes-dataset.com/.

## Inference on cityscapes

To run inference, this directory path needs to be added to your pythonpath. Here is the command for this:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/this/dir"
```

Here are code snippets to run inference on the models shown above. These examples show usage with 8 GPUs. You could run the inference command with 1/2/4 GPUs by updating the nproc_per_node argument. 

Our pretrained InverseForm module can be downloaded from here and should be placed inside the directory `checkpoints/`. See usage below.

[distance_measures_regressor.pth](https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/distance_measures_regressor.pth)


* HRNet-18-IF
```bash
python -m torch.distributed.launch --nproc_per_node=8 experiment/validation.py --output_dir "/path/to/output/dir" --model_path "checkpoints/hrnet18_IF_checkpoint.pth" --has_edge True
```
* HRNet-16-Slim-IF
```bash
python -m torch.distributed.launch --nproc_per_node=8 experiment/validation.py --output_dir "/path/to/output/dir" --model_path "checkpoints/hr16s_4k_slim.pth" --hrnet_base "16" --arch "lighthrnet.HRNet16" --has_edge True
```
* OCRNet-48-IF
```bash
python -m torch.distributed.launch --nproc_per_node=8 experiment/validation.py --output_dir "/path/to/output/dir" --model_path checkpoints/hrnet48_OCR_IF_checkpoint.pth --arch "ocrnet.HRNet" --hrnet_base "48" --has_edge True
```
* HMS-OCRNet-48-IF
```bash
python -m torch.distributed.launch --nproc_per_node=8 experiment/validation.py --output_dir "/path/to/output/dir" --model_path checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
```

To remove the InverseForm operation during inference, simply run without the has_edge flag. You will notice no drop in performance as compared to running with the operation. 

## Acknowledgements:

This repository shares code with the following repositories:

* Hierarchical Multi-Scale Attention(HMS): 
https://github.com/NVIDIA/semantic-segmentation
* HRNet-OCR: https://github.com/HRNet/HRNet-Semantic-Segmentation

We would like to acknowledge the researchers who made these repositories open-source.

