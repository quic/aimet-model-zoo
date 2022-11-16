# TensorFlow SRGAN (Super Resolution)

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

## Package Dependencies
```python
pip install scikit-image==0.16.2
pip install mmcv==1.2.0
pip install tensorflow-gpu==2.4.0
```

### Setup Super-resolution repo
- Clone the [krasserm](https://github.com/krasserm/super-resolution) repo:  
  `git clone https://github.com/krasserm/super-resolution.git`  
  `cd super-resolution`
- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=<path to super-resolution repo>/super-resolution:$PYTHONPATH`  
  `export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo`  

## Dataset 
- Three benchmark datasets can be downloaded from here:
  - [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
  - [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
  - [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)

## Model Weights
- The original SRGAN model is available at:
  - [krasserm](https://github.com/krasserm/super-resolution)

## Usage
```bash
  python3 srgan_quanteval.py --dataset-path <path to dataset>
```
- We only support 4x super resolution on .png images. So make sure your high resolution images are 4x the dimension of your low resolution images. If you are using one of the benchmark datasets, please use images under `image_SRF_4` directory.
- We assume low and high resolution images are both present under the same directory, with images that follow the below naming conventions:
  - low resolution images will have file name suffix: `LR.png`
    - e.g. `people_LR.png`
  - high resolution images will have file name suffix: `HR.png`
    - e.g. `people_HR.png`
    
## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 16 bits, asymmetric quantization
- Model inputs are quantized
- Bias Correction and Cross Layer Equalization have been applied
