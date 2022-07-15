# PyTorch-DeepLabV3+

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further. This model was tested using AIMET version 1.21.0.

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
2. Run the following commands to rearrange the contents of the DeepLabV3+ repository alongside the Model Zoo repository
```
mv modeling ../aimet-model-zoo/zoo_torch/examples/
mv dataloaders ../aimet-model-zoo/zoo_torch/examples/
mv utils/metrics.py ../aimet-model-zoo/zoo_torch/examples/
mv mypath.py ../aimet-model-zoo/zoo_torch/examples/
```
3. Download Optimized DeepLabV3+ checkpoint from the [Releases](/../../releases) page to any location in your workspace
4. Change the path of validation dataset(such as Pascal dataset) in the mypath.py to point to the path where the dataset was downloaded.

## Obtain model checkpoints, dataset and configuration
- The original DeepLabV3+ checkpoint can be downloaded here:
  - https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view
- Pascal Dataset can be downloaded from here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.qualcomm.com/qualcomm-ai/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_deeplabv3+.py \
        --checkpoint-path   <path to optimized checkpoint directory to load from> \
        --encodings-path <Path to optimized encodings> \
		    --use_cuda <Use cuda or cpu> \
		    --input-shape <Model input shape for quantization>
		    --quant-scheme <Quant scheme to use for quantization>
		    --default-output-bw <Default output bitwidth for quantization> \
		    --default-param-bw <Default parameter bitwidth for quantization> \
		    --config-file <Quantsim configuration file>
        --num-classes <number of classes in a dataset> \
        --dataset <dataset to be used for evaluation>
```

## INT8 Quantization Configuration
- Weight quantization: 8 bits, per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- Tf_enhanced was used as quantization scheme
- Cross layer equalization and Adaround has been applied on optimized checkpoint
- Data Free Quantization has been performed on the optimized checkpoint
