# MobileDet-EdgeTPU

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

*****

## Experiment setup 
###
*****

## Additional Dependencies

|      Package      |
| :---------------: |
| tensorflow/models |
|    pycocotools    |
|      tf_slim      |


*****
### Install dependent pip packages:

```bash
pip install pycocotools
pip install --upgrade tf_slim
```

### Download and install protoc
*protoc* is a standalone binary for the Google protobuf compiler. 

+ we use *protoc* in the version 3.14.0, download protoc-3.14.0-linux-x86_64.zip to *ROOT_PATH*
+ install

```bash
cd ROOT_PATH
unzip protoc-3.14.0-linux-x86_64.zip
cd protoc-3.14.0-linux-x86_64/bin
chmod +x protoc
export PATH=/ROOT_PATH/protoc-3.14.0-linux-x86_64/bin:$PATH 
```

Also, you can take this [reference](http://google.github.io/proto-lens/installing-protoc.html) for automatically download and installation. 

### Clone TensorFlow model zoo as the FP32 source

```bash
git clone https://github.com/tensorflow/models.git
git checkout master
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

*****

## Download model checkpoint for AIMET optimization
MobileDet-EdgeTPU FP32 pretrained checkpoint used for AIMET quantization can be downloaded from the [Releases](/../../releases) page.

*****

## Dataset: MSCOCO in the tfrecord format

TFRecord format of COCO dataset is needed. There are two options for download and process MSCOCO dataset: 
- **Option 1:** If you want to download and process MSCOCO dataset, use [download_and_preprocess_mscoco.sh](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh) to download and convert coco dataset into TFRecord

```bash
cd models/research/object_detection/dataset_tools
./download_and_preprocess_mscoco.sh <mscoco_dir>
```

- **Option 2:** If COCO dataset is already available or you want to download COCO dataset separately
  - COCO dataset can be download here: [COCO](https://cocodataset.org/#download)
    - Please download the 2017 Version
  - [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) can be used to convert dataset into TFRecord
  
```
python object_detection/dataset_tools/create_coco_tf_record.py --logtostderr --include_masks --train_image_dir=./MSCOCO_PATH/images/train2017/ --val_image_dir=./MSCOCO_PATH/images/val2017/ --test_image_dir=./MSCOCO_PATH/images/test2017/ --train_annotations_file=./MSCOCO_PATH/annotations/instances_train2017.json --val_annotations_file=./MSCOCO_PATH/annotations/instances_val2017.json --testdev_annotations_file=./MSCOCO_PATH/annotations/image_info_test2017.json --output_dir=./OUTPUT_DIR/
```
**Note:** The *--include_masks* option must be used. 

## Usage
- `mobiledet_edgetpu_quanteval.py` has two required arguments, an example usage is shown below
```bash
python mobiledet_edgetpu_quanteval.py --dataset-path <path to tfrecord dataset> --annotation-json-file <path to instances json file>/instances_val2017.json
```

*****

## Quantization configuration 
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:

+ Weight quantization: 8 bits, per-tensor symmetric quantization
+ Bias parameters are not quantized
+ Activation quantization: 8 bits, asymmetric quantization
+ Model inputs are not quantized
+ TF was used for weight quantization scheme
+ TF was used for activation quantization scheme
+ Weights are optimzied by per-tensor Adaround in TF_enhanced scheme
