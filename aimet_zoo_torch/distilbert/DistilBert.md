# PyTorch Transformer model DistilBert-base-uncased for Natural Language Classifier and Question Answering  
This document describes evaluation of optimized checkpoints for transformer models DistilBert-base-uncased for NL Classification tasks and Question Answering tasks.

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
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
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
### To run evaluation with QuantSim for Natural Language Classifier tasks in AIMET, use the following
```bash
python transformers_nlclassifier_quanteval.py \
        --model_name_or_path <MODEL_NAME> \
        --task_name <TASK> \
        --per_device_eval_batch_size 4 \
        --output_dir <OUT_DIR> \
```
* example 
    ```
    python transformers_nlclassifier_quanteval.py  --model_name_or_path distilbert-base-uncased   --task_name rte  --per_device_eval_batch_size 4 --output_dir ./evaluation_result 
    ```

* supported keyword of task_name supported are "rte","stsb","mrpc","cola","sst2","qnli","qqp","mnli"

* supported model_name_or_path are "bert-base-uncased", "google/mobilebert-uncased", "microsoft/DistilBert-base-uncased", "distilbert-base-uncased", "roberta-base"

### To run evaluation with QuantSim for Question Answering tasks in AIMET, use the following

```bash

python transformers_qa_quanteval.py \
    --model_name_or_path <MODEL_NAME> \
    --dataset_name <DATASET_NAME> \
    --per_device_eval_batch_size 4 \
    --output_dir <OUT_DIR>
```

* example
  ```
  python transformers_qa_quanteval.py  --model_name_or_path distilbert-base-uncased --dataset_name squad  --per_device_eval_batch_size 4 --output_dir ./evaluation_result 
  ```

* supported model_name_or_path are "bert-base-uncased", "google/mobilebert-uncased", "microsoft/DistilBert-base-uncased", "distilbert-base-uncased", "roberta-base"

* supported dataset_name is "squad"

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
Below are the results of the Pytorch transformer model DistilBert for GLUE dataset:

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
    <td> 53.85 </td>
    <td> 91.17 </td>
    <td> 88.40 </td>
    <td> 87.12 </td>
    <td> 90.39 </td>
    <td> 87.29 </td>
    <td> 82.15 </td>
    <td> 65.34 </td>
    <td> 80.71 </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> 52.99 </td>
    <td> 90.48 </td>
    <td> 89.34 </td>
    <td> 86.76 </td>
    <td> 89.77 </td>
    <td> 86.88 </td>
    <td> 83.35 </td>
    <td> 65.54 </td>
    <td> 80.26 </td>
  </tr>
</table>
