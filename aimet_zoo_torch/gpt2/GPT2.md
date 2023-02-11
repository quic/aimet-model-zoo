# PyTorch Transformer model GPT2 for Natural Language Text Generation 
This document describes evaluation of optimized checkpoints for transformer models GPT2 for NL Text Generation tasks.

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.24/packaging/install.md) (*Torch GPU* variant) before proceeding further.

**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.24.0*  (i.e. set `release_tag="1.24.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```
pip install accelerate==0.9.0
pip install transformers==4.21.0
pip install datasets==2.4.0

```
## Model checkpoint
- Original full precision checkpoints without downstream training were downloaded through hugging face 
- [Full precision model with downstream training weight files] are automatically downloaded using evaluation script 
- [Quantization optimized model weight files] are automatically downloaded using evaluation script 

## Dataset 
- For Text Generation tasks, we use the [ WikiText language modeling dataset](https://huggingface.co/datasets/wikitext) benchmark dataset for evaluation. 
- Dataset downloading is handled by evaluation script

## Usage
### To run evaluation with QuantSim for Natural Language Text Generation tasks in AIMET, use the following
```bash
python gpt2_quanteval.py \ 
    --model-config <model configuration > \
    --per_device_eval_batch_size <batch size> 

```
* Available configurations is : "gpt2_w8a8"
* example 
    ```
    python gpt2_quanteval.py --model_config gpt2_w8a8 --per_device_eval_batch_size 8  
    ```

## Quantization Configuration
The following configuration has been used for the above models for INT8 quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF range learning  was used as quantization scheme
- Clamped initialization was adopted
- Quantization aware training (QAT) was used to obtain optimized quantized weights
