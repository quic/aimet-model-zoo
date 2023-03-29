# Tensorflow SSD MobileNet v2

## Environment Setup 

### Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.25/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.25 for TensorFlow 2.4](https://github.com/quic/aimet/releases/tag/1.25) i.e. please set `release_tag="1.25"` and `AIMET_VARIANT="tf_gpu"` in the above instructions.

### Additional dependencies:
```bash
pip install pycocotools
pip install --upgrade tf_slim
pip install numpy==1.19.5
```

### Append the repo location to your `PYTHONPATH` by doing the following:
  `export PYTHONPATH=$PYTHONPATH:/<path to parent>/aimet-model-zoo`

### Dataset 
TFRecord format of 2017 COCO dataset is needed. There are two options for downloading and processing MSCOCO dataset: 
- **Option 1:** If you want to download and process MSCOCO dataset, use [download_and_preprocess_mscoco.sh](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh) to download and convert coco dataset into TFRecord:
```bash
cd models/research/object_detection/dataset_tools
./download_and_preprocess_mscoco.sh <mscoco_dir>
```

- **Option 2:** If COCO dataset is already available or you want to download COCO dataset separately
  - 2017 COCO dataset can be downloaded from here: [COCO](https://cocodataset.org/#download)
  - [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) can be used to convert dataset into TFRecord
  
```bash
python object_detection/dataset_tools/create_coco_tf_record.py --logtostderr --include_masks --train_image_dir=./MSCOCO_PATH/images/train2017/ --val_image_dir=./MSCOCO_PATH/images/val2017/ --test_image_dir=./MSCOCO_PATH/images/test2017/ --train_annotations_file=./MSCOCO_PATH/annotations/instances_train2017.json --val_annotations_file=./MSCOCO_PATH/annotations/instances_val2017.json --testdev_annotations_file=./MSCOCO_PATH/annotations/image_info_test2017.json --output_dir=./OUTPUT_DIR/
```
**Note:** The *--include_masks* option must be used. 

---

## Model checkpoint for AIMET optimization
 - Downloading of model checkpoints is handled by evaluation script.
 - Checkpoint used for AIMET quantization can be downloaded from the [Releases](/../../releases) page.

 ---

## Usage
```bash
python aimet_zoo_tensorflow/ssd_mobilenet_v2/evaluators/ssd_mobilenet_v2_quanteval.py \ 
 --model-config <model configuration to test> \ 
 --dataset-path <path to tfrecord dataset> \
 --annotation-json-file <path to instances json file>/instances_val2017.json
```

Supported model configurations are:
- ssd_mobilenetv2_w8a8

---

## Quantization configuration
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
