# Pytorch SalsaNext for LiDAR semantic segmentation

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.0

## Experiment setup
- If `aimet-model-zoo` source code is not ready, clone the [aimet-model-zoo](https://github.com/quic/aimet-model-zoo.git) repo 
```bash
  git clone https://github.com/quic/aimet-model-zoo.git
  cd ./aimet-model-zoo/aimet_zoo_torch/salsaNext/evaluators
```

- Clone the [salsaNext](https://github.com/TiagoCortinhal/SalsaNext.git) repo 
```bash
  git clone --recursive https://github.com/TiagoCortinhal/SalsaNext.git
  cd SalsaNext
```

- Create folders for log/output and pretrained model
```bash
  mkdir -p logs/sequences/08
  mkdir pretrained
```

- Prepare the AIMET code in salsaNext
```bash
  cp ../evaluation_func.py ./
  cp ../salsaNext_quanteval.py ./
```

- Update the salsaNext source code to adapt the AIMET quantization
```bash
  cp ../../models/SalsaNext.py ./train/tasks/semantic/modules/
```
   
## Model checkpoints and configuration
- Downloading checkpoints is handled through evaluation script. Configuration is set to default by evaluation script.
- The model checkpoints can be downloaded from
  - FP32 checkpoint `SalsaNext` [FP32](https://drive.google.com/file/d/10fxIwPK10UVVB9jsgXDZSDwj4vy9MyTl/view).
	- The corresponding `arch_cfg.yaml` and `data_cfg.yaml` are also downloaded from the above link.
  - Quantized files W8A8 `SalsaNext_optimized_model.pth` and `SalsaNext_optimized_encoding.encodings` [INT8](https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/releases/tag/torch_salsanext_models).  
- The Quantization Simulation (*Quantsim*) Configuration file `htp_quantsim_config_pt_pertensor.json` can be downloaded from [JSON](https://github.qualcomm.com/qualcomm-ai/aimet-model-zoo/releases/download/torch_salsanext_models/htp_quantsim_config_pt_pertensor.json). (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).
- These files should be arranged in the `./pretrained` folder with the following way
```
  <aimet_model_zoo_path>/SalsaNext/pretrained/
  ├── SalsaNext
  ├── SalsaNext_optimized_encoding.encodings
  ├── SalsaNext_optimized_model.pth
  ├── arch_cfg.yaml
  ├── data_cfg.yaml
  ├── htp_quantsim_config_pt_pertensor.json
```

## Dataset 
- Semantic-kitti dataset can be downloaded from here:
  - (http://semantic-kitti.org/tasks.html#semseg)
  
- Downloaded datasets should be arranged in one directory <dataset_path>
  - The <dataset_path> should be arranged in the following way
```
  <dataset_path_to_semnantic_kitti>/sequences/
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
	--dataset <path to dataset folder> \ 
	--log <path to log and output folder, e.g., full path of the created folder logs> \
	--model <path to pretrained model folder, e.g., full path of the created folder pretrained>
```
- One example
```bash
python salsaNext_quanteval.py --dataset <dataset_path_to_semnantic_kitti> --log ./logs/ --model ./pretrained/
```
## Quantization configuration 
- Weight quantization: 8 bits
- Activation quantization: 8 bits
- PTQ techniques: 
  - Firstly, apply batch_norm_fold API to make the folding, by `batch_norm_fold.fold_all_batch_norms`
  - Secondly, apply the Adaround API to optimize the weight, by `AdaroundParameters(*)` and `Adaround.apply_adaround(*)`
  - Finally, set the percentile (99.9%) as the quant scheme, by `sim.set_percentile_value(99.9)`
- The checkpoint is with 3 activation output with 16 bitwidth with QAT. 
  - `sim.model.downCntx.conv1.input_quantizers[0].bitwidth = 16`
  - `sim.model.resBlock1.pool.output_quantizers[0].bitwidth = 16`
  - `sim.model.module_softmax.output_quantizers[0].bitwidth = 16`  

## Results
<table style= " width:50%">
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2" style="text-align:center;">IoU avg</th>
    <th colspan="2" style="text-align:center;">Acc avg</th>
  </tr>
  <tr>
    <th>FP32</td>
    <th>INT8</td>
    <th>FP32</td>
    <th>INT8</td>
  </tr>
  <tr>
    <td rowspan="3">SalsaNext</td>
    <td>0.558</td>
    <td>0.549</td>
    <td>0.879</td>
    <td>0.874</td>
  </tr>
</table>

## FP32 results
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

## W8A8 results
```
Acc avg 0.874
IoU avg 0.549
IoU class 1 [car] = 0.863
IoU class 2 [bicycle] = 0.372
IoU class 3 [motorcycle] = 0.426
IoU class 4 [truck] = 0.780
IoU class 5 [other-vehicle] = 0.447
IoU class 6 [person] = 0.600
IoU class 7 [bicyclist] = 0.663
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.936
IoU class 10 [parking] = 0.395
IoU class 11 [sidewalk] = 0.789
IoU class 12 [other-ground] = 0.035
IoU class 13 [building] = 0.791
IoU class 14 [fence] = 0.444
IoU class 15 [vegetation] = 0.796
IoU class 16 [trunk] = 0.562
IoU class 17 [terrain] = 0.647
IoU class 18 [pole] = 0.457
IoU class 19 [traffic-sign] = 0.420
```
