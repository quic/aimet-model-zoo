# Pytorch SalsaNext for LiDAR semantic segmentation

## Environment Setup
### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.0

### Add AIMET Model Zoo to Pythonpath
- If `aimet-model-zoo` source code is not ready, clone the [aimet-model-zoo](https://github.com/quic/aimet-model-zoo.git) repo 
```bash
  git clone https://github.com/quic/aimet-model-zoo.git
  export PYTHONPATH=$PYTHONPATH:<path/to/aimet-model-zoo>
```
   
### Dataset 
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

---

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python salsanext_quanteval.py \
  --model-config <model configuration to use>
	--dataset-path <path to dataset folder> \ 
```
- Example
```bash
python salsaNext_quanteval.py --model-config salsanext_w8a8 --dataset-path <dataset_path_to_semnantic_kitti>
```
Or 
```bash
python salsaNext_quanteval.py --model-config salsanext_w4a8 --dataset-path <dataset_path_to_semnantic_kitti>
```

Available model configurations are:
- salsanext_w8a8
- salsanext_w4a8

---

## Model checkpoints and configuration
- Downloading checkpoints is handled through evaluation script. Configuration is set to default by evaluation script.
- The model checkpoints can be downloaded from
  - FP32 checkpoint `SalsaNext` [FP32](https://drive.google.com/file/d/10fxIwPK10UVVB9jsgXDZSDwj4vy9MyTl/view).
  - Quantized files W8A8 `SalsaNext_optimized_model.pth` and `SalsaNext_optimized_encoding.encodings` [W8A8](https://github.com/quic/aimet-model-zoo/releases/tag/torch_salsanext_models).  
  - Quantized files W4A8 `SalsaNext_optimized_w4A8_model.pth` and `SalsaNext_optimized_w4A8_encoding.encodings` [W4A8](https://github.com/quic/aimet-model-zoo/releases/tag/torch_salsanext_models).  
- The Quantization Simulation (*Quantsim*) Configuration file `default_config.json` can be downloaded from [W4A8_JSON](https://raw.githubusercontent.com/quic/aimet/release-aimet-1.25/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) and [W8A8_JSON](https://raw.githubusercontent.com/quic/aimet/develop/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json). (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

---

## Quantization configuration for W8A8
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

## Quantization configuration for W4A8
- Weight quantization: 4 bits
- Activation quantization: 8 bits
- PTQ techniques: 
  - Set the QuantScheme with `QuantScheme.training_range_learning_with_tf_init`
  - Apply batch_norm_fold API to make the folding, by `batch_norm_fold.fold_all_batch_norms`
- The checkpoint is with 9 activation output with 16 bitwidth with QAT. 
  - `sim.model.downCntx.conv3.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx.conv2.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx.conv1.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx.conv1.input_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx.act2.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx2.act2.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx3.act2.output_quantizers[0].bitwidth = 16`
  - `sim.model.downCntx.act1.output_quantizers[0].bitwidth = 16`
  - `sim.model.module_softmax.output_quantizers[0].bitwidth = 16` 
- The checkpoint is with 6 weight with 8 bitwidth with QAT.
  - `sim.model.downCntx.conv1.param_quantizers['weight'].bitwidth = 8`
  - `sim.model.downCntx.conv3.param_quantizers['weight'].bitwidth = 8`
  - `sim.model.downCntx2.conv3.param_quantizers['weight'].bitwidth = 8`
  - `sim.model.downCntx3.conv3.param_quantizers['weight'].bitwidth = 8`
  - `sim.model.downCntx.conv2.param_quantizers['weight'].bitwidth = 8`
  - `sim.model.downCntx.bn2.param_quantizers['weight'].bitwidth = 8`
---

## Results
<table style= " width:50%">
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3" style="text-align:center;">IoU avg</th>
    <th colspan="3" style="text-align:center;">Acc avg</th>
  </tr>
  <tr>
    <th>FP32</td>
    <th>W8A8</td>
	<th>W4A8</td>
    <th>FP32</td>
    <th>W8A8</td>
	<th>W4A8</td>
  </tr>
  <tr>
    <td rowspan="3">SalsaNext</td>
    <td>0.558</td>
    <td>0.549</td>
	<td>0.551</td>
    <td>0.879</td>
    <td>0.874</td>
	<td>0.876</td>
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

## W4A8 results
```
Acc avg 0.876
IoU avg 0.551
IoU class 1 [car] = 0.863
IoU class 2 [bicycle] = 0.381
IoU class 3 [motorcycle] = 0.437
IoU class 4 [truck] = 0.770
IoU class 5 [other-vehicle] = 0.428
IoU class 6 [person] = 0.606
IoU class 7 [bicyclist] = 0.670
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.937
IoU class 10 [parking] = 0.378
IoU class 11 [sidewalk] = 0.788
IoU class 12 [other-ground] = 0.053
IoU class 13 [building] = 0.798
IoU class 14 [fence] = 0.467
IoU class 15 [vegetation] = 0.800
IoU class 16 [trunk] = 0.567
IoU class 17 [terrain] = 0.646
IoU class 18 [pole] = 0.456
IoU class 19 [traffic-sign] = 0.428
