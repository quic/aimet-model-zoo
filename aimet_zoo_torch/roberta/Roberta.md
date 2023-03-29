# PyTorch Transformer model Roberta-base for Natural Language Classification   
This document describes evaluation of optimized checkpoints for transformer models Roberta-base for Natural Language tasks. 

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.23/packaging/install.md) (*Torch GPU* variant) before proceeding further.

**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.23.0*  (i.e. set `release_tag="1.23.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```
pip install datasets==2.4.0
pip install transformers==4.11.3
```
## Model checkpoint
- Original full precision checkpoints without downstream training were downloaded through hugging face 
- [Full precision model with downstream training weight files] are automatically downloaded using evaluation script 
- [Quantization optimized model weight files] are automatically downloaded using evaluation script 

## Dataset 
- For NLP tasks, we use the [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) benchmark dataset for evaluation. 
- For Question Answering tasks, we use the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer) benchmark dataset for evaluation. 
- Dataset downloading is handled by evaluation script

## Usage
### To run evaluation with QuantSim in AIMET, use the following
```bash
python roberta_quanteval.py \
        --model_config <MODEL_CONFIGURATION> \
        --per_device_eval_batch_size 4 \
        --output_dir <OUT_DIR> \
```
* example 
    ```
    python roberta_quanteval.py --model_config roberta_w8a8_rte --per_device_eval_batch_size 4 --output_dir ./evaluation_result 
    ```

* supported values of model_config are "roberta_w8a8_rte","roberta_w8a8_stsb","roberta_w8a8_mrpc","roberta_w8a8_cola","roberta_w8a8_sst2","roberta_w8a8_qnli","roberta_w8a8_qqp","roberta_w8a8_mnli"


## Quantization Configuration
The following configuration has been used for the above models for INT8 quantization:
- Weight quantization: 8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF range learning  was used as quantization scheme
- Mask values of -6 was applied in attention layers
- Quantization aware training (QAT) was used to obtain optimized quantized weights, detailed hyperparameters listed in [Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort, "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", EMNLP 2021](https://arxiv.org/abs/2109.12948).

## Results
Below are the results of the Pytorch transformer model Roberta for GLUE dataset:

<table style= " width:50%">
  <tr>
    <td> Configuration </td>
    <td> CoLA (corr) </td>
    <td> SST-2 (acc) </td>
    <td> MRPC (f1) </td>
    <td> STS-B (corr) </td>
    <td> QQP (acc) </td>
    <td> MNLI (acc) </td>
    <td> QNLI (acc) </td>
    <td> RTE (acc) </td>
    <td> GLUE </td>
  </tr>
  <tr>
    <td> FP32 </td>
    <td> 60.36 </td>
    <td> 94.72 </td>
    <td> 91.84 </td>
    <td> 90.54 </td>
    <td> 91.24 </td>
    <td> 87.29 </td>
    <td> 92.33 </td>
    <td> 72.56 </td>
    <td> 85.11 </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> 57.35 </td>
    <td> 92.55 </td>
    <td> 92.69 </td>
    <td> 90.15 </td>
    <td> 90.09 </td>
    <td> 86.88 </td>
    <td> 91.47 </td>
    <td> 72.92 </td>
    <td> 84.26 </td>
  </tr>
</table>
