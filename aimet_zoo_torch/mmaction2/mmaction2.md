# PyTorch mmaction2

## Environment Setup

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.25.


### Install dependencies 
```bash 
python -m pip install gdown
```

Please follow the steps from [open-mmlab/mmaction2 install guide](https://github.com/open-mmlab/mmaction2#%EF%B8%8F-installation-)
to install mmaction2 as dependency. The package versions we used for open-mmlab are:
- mmaction2 1.0.0
- mmengine 0.7.3
- mmcv 2.0.0

Append the repo location to your `PYTHONPATH` with the following:  
```bash
export PYTHONPATH=$PYTHONPATH:<path to aimet-model-zoo>
```

### Dataset
Instructions to prepare ActivityNet can be found at:
- https://github.com/open-mmlab/mmaction2/blob/main/tools/data/activitynet/README.md
Note that option 1 was used for this model

After downloading and processing the dataset, please change the data path to point to your download location in
aimet_zoo_torch/mmaction2/model/configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py

---

## Usage
Before running the evaluation script, set your config path in the model cards via replacing with your own path in the 
"config" field. The model cards are .json files located under model/model_cards/

To run evaluation with QuantSim in AIMET, use the following
```bash
python  aimet_zoo_torch/mmaction2/evaluators/mmaction2_quanteval.py --model-config <configuration to be tested> --use-cuda
```

Available model configurations are:
- bmn_w8a8


---

## Quantization Configuration
- Weight quantization: 8 bits, per tensor symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF enhanced was used as quantization scheme
