# PyTorch DeepSpeech

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Setup DeepSpeech2
- Clone the [SeanNaren DeepSpeech2 Repo](https://github.com/SeanNaren/deepspeech.pytorch)  
  `git clone https://github.com/SeanNaren/deepspeech.pytorch.git`

- checkout this commit id:  
`cd deepspeech.pytorch`  
`git checkout 78f7fb791f42c44c8a46f10e79adad796399892b`

- Append the repo locations to your `PYTHONPATH` with the following:  
```
export PYTHONPATH=$PYTHONPATH:<path to parent>/deepspeech.pytorch
export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo
```

- Install requirements :  
`pip install -r aimet-model-zoo/zoo_torch/examples/deepspeech2/requirements.txt`

## Obtain the Test Dataset

- The evaluation script will automatically download the model checkpoint from [here](https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth).

- Run the command below to [download](https://github.com/SeanNaren/deepspeech.pytorch/blob/v2.0/data/librispeech.py) the dataset and format the csv needed for the `test-manifest` flag.

```bash
python3 deepspeech.pytorch/data/librispeech.py --files-to-use test-clean.tar.gz
```

Details are available on the [Datasets Section in the SeanNaren Repo](https://github.com/SeanNaren/deepspeech.pytorch#datasets).

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python deepspeech2_quanteval.py \
  --test-manifest=<path to test manifest csv>
```

## Quantization Configuration
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are not quantized
- Model inputs are quantized
- Activation quantization: 8 bits, asymmetric quantization
  - Inputs to Conv layers are quantized
  - Input and recurrent activations for LSTM layers are quantized
- Quantization scheme is tf enhanced
- Operations which shuffle data such as reshape or transpose do not require additional quantizers
