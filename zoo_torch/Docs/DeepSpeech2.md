# DeepSpeech

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies

### Setup SeanNaren DeepSpeech2 Repo

- Clone the [SeanNaren DeepSpeech2 Repo](https://github.com/SeanNaren/deepspeech.pytorch)  
  `git clone https://github.com/SeanNaren/deepspeech.pytorch.git`

- checkout this commit id:  
`cd deepspeech.pytorch`  
`git checkout 78f7fb791f42c44c8a46f10e79adad796399892b`

- Install the requirements from the SeanNaren repo as detailed in the repository.

- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=$PYTHONPATH:<path to SeanNaren deepspeech2 repo>/deepspeech.pytorch`


## Obtaining model checkpoint and dataset

- The SeanNaren DeepSpeech2 checkpoint can be downloaded from [here](https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth). Please point the `model-path` flag in to this file in the run script. Please note that this script is only compatible with release V2.

- LibriSpeech __test clean__ set can be downloaded here:
  - http://www.openslr.org/12


Please see the [Datasets Section in the SeanNaren Repo](https://github.com/SeanNaren/deepspeech.pytorch#datasets) for the format of the test manifest used in the script. The [download script](https://github.com/SeanNaren/deepspeech.pytorch/blob/v2.0/data/librispeech.py) from this repository will download and format the csv to be used in the `test-manifest` flag.


## Usage

- To run evaluation with QuantSim in AIMET, use the following

```bash
python deepspeech2_quanteval.py \
  --model-path=<path to DeepSpeech2 checkpoint> \
  --test-manifest=<path to test manifest csv>
```

## Quantizer Op Assumptions
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
  - Inputs to Conv layers are quantized
  - Input and recurrent activations for LSTM layers are quantized
- Operations which shuffle data such as reshape or transpose do not require additional quantizers
