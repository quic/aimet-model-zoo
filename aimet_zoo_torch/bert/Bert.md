# PyTorch Transformer model Bert-base-uncased for Natural Language Classifier and Question Answering  
This document describes evaluation of optimized checkpoints for transformer models Bert-base-uncased for NL Classification tasks and Question Answering tasks.

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
python bert_quanteval.py \
        --model_config <MODEL_CONFIGURATION> \
        --per_device_eval_batch_size 4 \
        --output_dir <OUT_DIR> \
```
* example 
    ```
    python bert_quanteval.py --model_config bert_w8a8_rte  --per_device_eval_batch_size 4 --output_dir ./evaluation_result 
    ```

* supported values of model_config are "bert_w8a8_rte","bert_w8a8_stsb","bert_w8a8_mrpc","bert_w8a8_cola","bert_w8a8_sst2","bert_w8a8_qnli","bert_w8a8_qqp","bert_w8a8_mnli", "bert_w8a8_squad"


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
Below are the results of the Pytorch transformer model Bert for GLUE dataset:

<table style= " width:50%">
  <tr>
    <td> Configuration </td>
    <td> CoLA (corr)  </td>
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
    <td> 58.76 </td>
    <td> 93.12 </td>
    <td> 89.93 </td>
    <td> 88.84 </td>
    <td> 90.94 </td>
    <td> 85.19 </td>
    <td> 91.63 </td>
    <td> 66.43 </td>
    <td> 83.11 </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> 56.93 </td>
    <td> 91.28 </td>
    <td> 90.34 </td>
    <td> 89.13 </td>
    <td> 90.78 </td>
    <td> 81.68 </td>
    <td> 91.14 </td>
    <td> 68.23 </td>
    <td> 82.44 </td>
  </tr>
</table>
