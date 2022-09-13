# PyTorch HRNET-Posenet
This document describes evaluation of optimized checkpoint for Hrnet-posenet

## Workspace setup
Clone the AIMET Model Zoo repo into your workspace:  
`git clone https://github.com/quic/aimet-model-zoo.git`

## AIMET installation and setup
Install the *Torch GPU* variant of AIMET package *and* setup the environment using the instructions here:
https://github.com/quic/aimet/blob/develop/packaging/install.md

---
**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.21.0*  (i.e. set `release_tag="1.21.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```bash
sudo -H pip install yacs
sudo -H pip install json-tricks
sudo -H pip install pycocotools
sudo -H pip install Cython
sudo -H pip install opencv-python==3.4.1.15
sudo -H apt-get update
sudo -H apt-get install ffmpeg
sudo -H chmod 777 -R <path_to_python_package>/dist-packages/*

cd <path_to_aimet_modelzoo>/zoo_torch/examples
git clone https://github.com/HRNet/HRNet-Human-Pose-Estimation.git
cd HRNet-Human-Pose-Estimation/
git checkout 00d7bf72f56382165e504b10ff0dddb82dca6fd2
cp -r ./lib/ ../hrnet-posenet/

cd zoo_torch/examples/hrnet-posenet/lib
make
```

## Modifications
Add the following lines inside `./hrnet-posenet/lib/core/function.py`

Addition at line 105
```
on_cuda = next(model.parameters()).is_cuda
```

Addition at line 121
```
if on_cuda:
	input=input.cuda()
```

## Obtaining model checkpoint and dataset
- FP32 and Optimized checkpoint of HRNET-posenet can be downloaded from the [Releases](/../../releases) page.
- COCO dataset can be downloaded from here:
  - [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip)
  - [COCO 2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
cd <path_to_aimet_modelzoo>/zoo_torch/examples/hrnet-posenet
python hrnet_posenet_quanteval.py
	--default-param-bw <weight bitwidth for quantization - 8 for INT8> \
	--default-output-bw <output bitwidth for quantization - 8 for INT8> \
	--use-cuda <boolean for using cuda> \
	--evaluation-dataset <path to MS-COCO validation dataset>

eg.
python hrnet_posenet_quanteval.py --default-param-bw=8 --default-output-bw=8 --use-cuda=True --evaluation-dataset=<path_to_MSCOCO_mainDIR>
```

## Quantization Configuration
The following configuration has been used for the above model for INT8 quantization

- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- 320 images (10 batches) from the validation dataloader was used for compute encodings
- Batchnorm folding and "TF" quantscheme in per channel mode has been applied to get the INT8 optimized checkpoint
