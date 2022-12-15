# Tensorflow SSD MobileNet v2

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

## Experiment Setup 

### Additional dependencies:
```bash
pip install pycocotools
pip install --upgrade tf_slim
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

### Download and install protoc
*protoc* is a standalone binary for the Google protobuf compiler. 
+ we use *protoc* in the version 3.14.0, use the following commands to download and install protoc-3.14.0-linux-x86_64.
```bash
PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
unzip protoc-3.14.0-linux-x86_64.zip
export PATH=<path to protoc>/protoc-3.14.0-linux-x86_64/bin:$PATH 
```

### Clone TensorFlow model zoo as FP32 source
```bash
git clone https://github.com/tensorflow/models.git
git checkout master
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

### Append the repo location to your `PYTHONPATH` by doing the following:
  `export PYTHONPATH=<path to tensorflow models repo>/models/research:$PYTHONPATH`

## Model checkpoint for AIMET optimization
 - Downloading of model checkpoints is handled by evaluation script.
 - SSD MobileNet v2 checkpoint used for AIMET quantization can be downloaded from the [Releases](/../../releases) page.

## Dataset 
TFRecord format of 2017 COCO dataset is needed. There are two options for download and process MSCOCO dataset: 
- **Option 1:** If you want to download and process MSCOCO dataset, use [download_and_preprocess_mscoco.sh](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh) to download and convert coco dataset into TFRecord:
```bash
cd models/research/object_detection/dataset_tools
./download_and_preprocess_mscoco.sh <mscoco_dir>
```

- **Option 2:** If COCO dataset is already available or you want to download COCO dataset separately
  - 2017 COCO dataset can be download here: [COCO](https://cocodataset.org/#download)
  - [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) can be used to convert dataset into TFRecord
  
```bash
python object_detection/dataset_tools/create_coco_tf_record.py --logtostderr --include_masks --train_image_dir=./MSCOCO_PATH/images/train2017/ --val_image_dir=./MSCOCO_PATH/images/val2017/ --test_image_dir=./MSCOCO_PATH/images/test2017/ --train_annotations_file=./MSCOCO_PATH/annotations/instances_train2017.json --val_annotations_file=./MSCOCO_PATH/annotations/instances_val2017.json --testdev_annotations_file=./MSCOCO_PATH/annotations/image_info_test2017.json --output_dir=./OUTPUT_DIR/
```
**Note:** The *--include_masks* option must be used. 

## Usage
- `ssd_mobilenet_v2_quanteval.py` has two required arguments, an example usage is shown below
```bash
python ssd_mobiledet_v2_quanteval.py 
 --dataset-path <path to tfrecord dataset> \
 --annotation-json-file <path to instances json file>/instances_val2017.json \
 --model-to-eval < which model to evaluate.Two options are available: fp32 for evaluating original fp32 model, int8 for evaluating quantized int8 model.>
```

## Quantization configuration
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
