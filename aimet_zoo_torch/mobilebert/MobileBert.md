# PyTorch Transformer model Mobilebert-uncased for Natural Language Classification and Question Answering  
This document describes evaluation of optimized checkpoints for transformer models Mobilebert-uncased for NL Classification and Question Answering tasks.

## AIMET installation and setup
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.27/packaging/install.md) (*Torch GPU* variant) before proceeding further.

**NOTE**
- All AIMET releases are available here: https://github.com/quic/aimet/releases
- This model has been tested using AIMET version *1.27.0*  (i.e. set `release_tag="1.27.0"` in the above instructions).
- This model is compatible with the PyTorch GPU variant of AIMET (i.e. set `AIMET_VARIANT="torch_gpu"` in the above instructions).

## Additional Setup Dependencies
```
pip install datasets==2.4.0
pip install transformers==4.11.3 
```

## Add AIMET Model Zoo to the PYTHONPATH
```bash 
export PYTHONPATH=$PYTHONPATH:<path to parent of aimet_model_zoo_path>
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
python mobilebert_quanteval.py \
        --model_config <MODEL_CONFIGURATION> \
        --per_device_eval_batch_size 4 \
        --output_dir <OUT_DIR> \
```
* example 
    ```
    python mobilebert_quanteval.py --model_config mobilebert_w8a8_rte --per_device_eval_batch_size 4 --output_dir ./evaluation_result 
    ```

* supported values of model_config are "mobilebert_w8a8_rte","mobilebert_w8a8_stsb","mobilebert_w8a8_mrpc","mobilebert_w8a8_cola","mobilebert_w8a8_sst2","mobilebert_w8a8_qnli","mobilebert_w8a8_qqp","mobilebert_w8a8_mnli", "mobilebert_w8a8_squad", "mobilebert_w4a8_rte","mobilebert_w4a8_stsb","mobilebert_w4a8_mrpc","mobilebert_w4a8_cola","mobilebert_w4a8_sst2","mobilebert_w4a8_qnli","mobilebert_w4a8_qqp","mobilebert_w4a8_mnli", "mobilebert_w4a8_squad"

## Quantization Configuration
The following configuration has been used for the above models for IN4/INT8 quantization:
- Weight quantization: 4/8 bits, symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- Different quantization scheme for different weight bithwidth and downstreaming tasks
- Mask values of -6 was applied in attention layers
- Quantization aware training (QAT) was used to obtain optimized quantized weights, detailed hyperparameters listed in [Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort, "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", EMNLP 2021](https://arxiv.org/abs/2109.12948).

<table style= " width:50%">
  <tr>
    <td> QAT Configuration </td>
    <td> CoLA </td>
    <td> SST-2 </td>
    <td> MRPC </td>
    <td> STS-B </td>
    <td> QQP </td>
    <td> MNLI </td>
    <td> QNLI </td>
    <td> RTE </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> per-tensor, tf </td>
    <td> per-channel, tf </td>
    <td> per-channel, tf </td>
    <td> per-channel, tf </td>
    <td> per-channel, tf </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-channel, tf </td>
  </tr>
  <tr>
    <td> W4A8 </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-channel, tf_enhanced </td>
    <td> per-channel, tf_enhanced </td>
    <td> per-channel, tf_enhanced </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-tensor, tf_enhanced </td>
    <td> per-channel, tf_enhanced </td>
  </tr>
</table>

<table style= " width:50%">
  <tr>
    <td> QAT Configuration </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> per-channel, tf </td>
  </tr>
  <tr>
    <td> W4A8 </td>
    <td> per-channel, range_learning_with_tf_enhanced_init </td>
  </tr>
</table>

## Results
Below are the results of the Pytorch transformer model MobileBert for GLUE dataset:

<table style= " width:50%">
  <tr>
    <td>  </td>
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
    <td> 51.48 </td>
    <td> 91.60 </td>
    <td> 85.86 </td>
    <td> 88.22 </td>
    <td> 90.66 </td>
    <td> 83.54 </td>
    <td> 91.18 </td>
    <td> 68.60 </td>
    <td> 81.27 </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> 52.51 </td>
    <td> 91.63 </td>
    <td> 90.81 </td>
    <td> 88.19 </td>
    <td> 90.80 </td>
    <td> 83.46 </td>
    <td> 91.12 </td>
    <td> 68.95 </td>
    <td> 82.18 </td>
  </tr>
  <tr>
    <td> W4A8 </td>
    <td> 50.34 </td>
    <td> 91.28 </td>
    <td> 87.61 </td>
    <td> 87.30 </td>
    <td> 90.48 </td>
    <td> 82.90 </td>
    <td> 89.42 </td>
    <td> 68.23 </td>
    <td> 80.95 </td>
  </tr>
</table>

<table style= " width:50%">
  <tr>
    <td>  </td>
    <td> EM </td>
    <td> F1 </td>
  </tr>
  <tr>
    <td> FP32 </td>
    <td> 82.75 </td>
    <td> 90.11 </td>
  </tr>
  <tr>
    <td> W8A8 </td>
    <td> 81.96 </td>
    <td> 89.41 </td>
  </tr>
  <tr>
    <td> W4A8 </td>
    <td> 81.88 </td>
    <td> 89.33 </td>
  </tr>
</table>
