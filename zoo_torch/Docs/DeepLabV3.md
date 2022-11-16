# PyTorch-DeepLabV3+

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Additional Dependencies
1. Install pycocotools as follows
```
sudo -H pip install pycocotools
```

## Model modifications & Experiment Setup
1. Clone the [DeepLabV3+ repo](https://github.com/jfzhang95/pytorch-deeplab-xception)
```
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd pytorch-deeplab-xception
git checkout 9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6
```
2. Add DeepLabV3+ and the model zoo repo to the Python path
```
export PYTHONPATH=$PYTHONPATH:<path to parent>/pytorch-deeplab-xception
export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo
```

## Dataset 
The Pascal Dataset can be downloaded from here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

In the `pytorch-deeplab-xception/mypath.py` script, change the Pascal dataset path to point to the path where the dataset was downloaded.

## Model checkpoint and configuration
Ensure you have gdown to obtain the model's weights:  
```
pip install gdown
 ```

- The original DeepLabV3+ checkpoint can be downloaded from here:
  - https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python deeplabv3_quanteval.py \
		--use_cuda <Use cuda or cpu, default True> \
		--batch-size <Number of images per batch, default 4>
```

## Quantization Configuration
INT8 optimization
The following configuration has been used for the above model for INT8 quantization
- Weight quantization: 8 bits, per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround has been applied on optimized checkpoint
- Data Free Quantization has been performed on the optimized checkpoint

INT4 optimization
The following configuration has been used for the above model for W4A8 quantization
- Weight quantization: 4 bits, per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF-Enhanced was used as quantization scheme
- Cross layer equalization and Adaround has been applied on optimized checkpoint
- Data Free Quantization has been performed on the optimized checkpoint
- Quantization Aware Traning has been performed on the optimized checkpoint
