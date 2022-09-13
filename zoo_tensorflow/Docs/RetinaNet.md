# TensorFlow RetinaNet

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

## Environment Requirements
This model requires the following python package versions:  
```
pip install tensorflow-gpu==1.15.0  
pip install keras==2.2.4
pip install progressbar2>=4.0.0
pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

It also required libGL1:
```
sudo apt update
sudo apt-get install libgl1 -y
```

Note that this model is expected **not** to work with GPUs at or after NVIDIA 30-series (e.g. RTX 3050), as those bring a new architecture not fully compatible with TF 1.X

### Setup RetinaNet Repo
- Clone the RetinaNet repository from github: https://github.com/fizyr/keras-retinanet  
```git clone https://github.com/fizyr/keras-retinanet.git```  
```cd keras-retinanet```

- Within the cloned repository, checkout the commit corresponding to pre-tf2.0. The included example scripts only works for TF 1.x.  
  ```git checkout 08af308d01a8f22dc286d62bc26c8496e1ff6539```

- Install keras-retinanet and dependencies using by running:  
  ```pip install . --user```

## Add AIMET Model Zoo to the Python Path
`export PYTHONPATH=$PYTHONPATH:<path to parent>/aimet-model-zoo`

## Dataset
- The COCO dataset can be downloaded from here:
  - http://cocodataset.org

## Model Weights
- The original pre-trained keras retinanet model is available here:
  - [RetinaNet pre-trained model](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5)

## Usage
The evaluation script supports 4 actions: evaluating the original model on GPU ("original_fp32"); evaluating the original model on a simulated device ("original_int8");
evaluating the optimized model on GPU ("optimized_fp32"); evaluating the optimized model on a simulated device ("optimized_int8").
```
python3 retinanet_quanteval.py \
        --dataset-Path <path to location of coco dataset> \
        --action <one of: original_fp32, original_int8, optimized_fp32, optimized_int8>
```

## Quantization Configuration
- Weight quantization: 8 bits, per tensor asymmetric quantization
- Bias parameters are quantized
- Activation quantization: 8 bits, asymmetric quantization
