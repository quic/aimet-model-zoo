# Pytorch SRGAN (Super Resolution)

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Experiment setup
- Clone the [mmsr](https://github.com/andreas128/mmsr) repo and apply patch  
```bash
  git clone https://github.com/andreas128/mmsr.git
  cd mmsr
  git checkout a73b318f0f07feb6505ef5cb1abf0db33e33807a
  git apply ../aimet-model-zoo/zoo_torch/examples/srgan/srgan_eval.patch
```
- Install dependencies 
```bash 
   python -m pip install lmdb
```
- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=<path to mmsr repo>:<path to mmsr repo>/codes:$PYTHONPATH`

  Note that here we add both `mmsr` and the subdirectory `mmsr/codes` to our path.
- Loading AIMET model zoo libraries  
`export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>`
   
## Model checkpoints and configuration
- Downloading checkpoints is handled through evaluation script. Configuration is set to default by evaluation script.
- The SRGAN model checkpoints can be downloaded from [mmediting](/../../releases/tag/srgan_mmsr_model).
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json). (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset 
- Three benchmark datasets can be downloaded from here:
  - [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
  - [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
  - [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
  
  Our benchmark results use images under **image_SRF_4** directory which tests 4x super-resolution as the suffix number indicates. You can also use other scales. See instructions for usage below.
- Downloaded datasets should be arranged in one directory <dataset_path>
  - The <dataset_path> should be arranged in the following way
```
  <dataset_path>/
  ├── Set5
  │   ├── image_SRF_2
  │   ├── image_SRF_3
  │   ├── image_SRF_4
  ├── Set14
  │   ├── image_SRF_2
  │   ├── image_SRF_3
  │   ├── image_SRF_4
  ├── BSD100
  │   ├── image_SRF_2
  │   ├── image_SRF_3
  │   ├── image_SRF_4
```

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python srgan_quanteval.py \
	--mmsr-path <path to patched mmsr git repo> \
	--dataset-path <path to dataset folder> \
	--use-cuda <Run evaluation on GPU> \
	--output-dir <path to output images>
```

## Quantization configuration 
- Weight quantization: 8 bits per tensor asymmetric quantization
- Bias parameters are quantized
- Activation quantization: 8 bits asymmetric quantization
- Model inputs are not quantized

## Results
<table style= " width:50%">
   <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>PSNR</th>
    <th>SSIM</th>
  </tr>
  <tr>
    <td>FP32</td>
    <td>Set5 / Set14 / BSD100</td>
    <td>29.93 / 26.58 / 25.51</td>
    <td>0.851 / 0.709 / 0.653</td>
  </tr>
  <tr>
    <td>INT8</td>
    <td>Set5 / Set14 / BSD100</td>
    <td>29.86 / 26.59 / 25.55</td>
    <td>0.845 / 0.705 / 0.648</td>
  </tr>
</table>
