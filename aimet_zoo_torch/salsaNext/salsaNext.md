# Pytorch SalsaNext for lidar semantic segmentation

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.24.0

## Experiment setup
- Clone the [salsaNext](https://github.com/TiagoCortinhal/SalsaNext.git) repo 
```bash
  git clone --recursive https://github.com/TiagoCortinhal/SalsaNext.git
```

- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=<path to salsaNext repo>:<path to salsaNext repo>/train:$PYTHONPATH`

- Loading AIMET model zoo libraries  
`export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>`

## Source code change to meet the model preparer
- Replace the file ./SalsaNext/train/tasks/semantic/modules/SalsaNext.py, with aimet_zoo_torch/salsaNext/models/SalsaNext.py
- For AIMET compatibility, the provided file has the following modifications to `class UpBlock(nn.Module)`:
  - we add `self.pixel_shuffle = nn.PixelShuffle(2)`
  - we replace `upA = nn.PixelShuffle(2)(x)` with `upA = self.pixel_shuffle(x)`
   
## Model checkpoints and configuration
- Downloading checkpoints is handled through evaluation script. Configuration is set to default by evaluation script.
- The salsaNext model checkpoints can be downloaded from
  - FP32 checkpoint [FP32](https://drive.google.com/file/d/10fxIwPK10UVVB9jsgXDZSDwj4vy9MyTl/view).
  - Quantized files W8A8 [INT8]((https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/releases)).
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json). (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset 
- Semantic-kitti dataset can be downloaded from here:
  - (http://semantic-kitti.org/tasks.html#semseg)
  
- Downloaded datasets should be arranged in one directory <dataset_path>
  - The <dataset_path> should be arranged in the following way
```
  <dataset_path>/sequences/
  ├── 00
  │   ├── labels/
  │   ├── velodyne/
  │   ├── calib.txt
  │   ├── poses.txt
  │   ├── times.txt  
  ├── 01
  │   ├── labels/
  │   ├── velodyne/
  │   ├── calib.txt
  │   ├── poses.txt
  │   ├── times.txt
```

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python salsaNext_quanteval.py \
	-d <path to dataset folder> \ 
	-l <path to log and output folder> \
	-m <path to pretrained model folder>
```

## Quantization configuration 
- Weight quantization: 8 bits per channel quantization
- Activation quantization: 8 bits
- PTQ techniques: 
  - Firstly, apply batch_norm_fold API to make the folding, by `batch_norm_fold.fold_all_batch_norms`
  - Secondly, apply the Adaround API to optimize the weight, by `AdaroundParameters(*)` and `Adaround.apply_adaround(*)`
  - Finally, set the percentile (99.9%) as the quant scheme, by `sim.set_percentile_value(99.9)`
- Enable one activation output with 16 bitwidth. 
  - `sim.model.downCntx.conv1.input_quantizers[0].bitwidth = 16`

## FP32 Results
```
Acc avg 0.879
IoU avg 0.558
IoU class 1 [car] = 0.862
IoU class 2 [bicycle] = 0.394
IoU class 3 [motorcycle] = 0.420
IoU class 4 [truck] = 0.777
IoU class 5 [other-vehicle] = 0.420
IoU class 6 [person] = 0.621
IoU class 7 [bicyclist] = 0.683
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.943
IoU class 10 [parking] = 0.422
IoU class 11 [sidewalk] = 0.800
IoU class 12 [other-ground] = 0.041
IoU class 13 [building] = 0.800
IoU class 14 [fence] = 0.484
IoU class 15 [vegetation] = 0.803
IoU class 16 [trunk] = 0.579
IoU class 17 [terrain] = 0.642
IoU class 18 [pole] = 0.466
IoU class 19 [traffic-sign] = 0.445
```

## W8A8 Results
```
Acc avg 0.887
IoU avg 0.554
IoU class 1 [car] = 0.909
IoU class 2 [bicycle] = 0.345
IoU class 3 [motorcycle] = 0.363
IoU class 4 [truck] = 0.665
IoU class 5 [other-vehicle] = 0.417
IoU class 6 [person] = 0.597
IoU class 7 [bicyclist] = 0.670
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.926
IoU class 10 [parking] = 0.428
IoU class 11 [sidewalk] = 0.767
IoU class 12 [other-ground] = 0.019
IoU class 13 [building] = 0.850
IoU class 14 [fence] = 0.499
IoU class 15 [vegetation] = 0.830
IoU class 16 [trunk] = 0.606
IoU class 17 [terrain] = 0.653
IoU class 18 [pole] = 0.580
IoU class 19 [traffic-sign] = 0.404
```
