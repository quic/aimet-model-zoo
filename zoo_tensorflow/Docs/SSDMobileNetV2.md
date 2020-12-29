# SSD MobileNet v2

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Obtaining model checkpoint and dataset

- SSD MobileNet v2 checkpoint used for AIMET quantization can be downloaded from release page
- Or you could follow the steps below to obtain the checkpoint

### export inference graph

The following steps are need to have a model ready for AIMET quantization

- ```bash
  git clone https://github.com/tensorflow/models.git
  cd models && git checkout r1.12.0
  cd research && protoc object_detection/protos/*.proto --python_out=.
  ```

- Download [ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz)

  - `tar xfvz ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz`

  - **remove** following parameters from `pipeline.config` that come with the tarball

    ```
    graph_rewriter {
          quantization {
          delay: 48000
          weight_bits: 8
          activation_bits: 8
        }  
    ```
- Add the following code snippet  to [Line 147, models/research/object_detection/export_inference_graph.py](https://github.com/tensorflow/models/blob/r1.12.0/research/object_detection/export_inference_graph.py#L147)
  
    ```python
    import os
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, os.path.join(FLAGS.output_directory, "model.ckpt"))
      aimet_model_output_dir = os.path.join(FLAGS.output_directory, "AIMET")
    os.mkdir(aimet_model_output_dir)
    saver.save(sess, os.path.join(aimet_model_output_dir, "model.ckpt")
	```
    
- tensorflow v1.10 is need to run the script, we could use the offical tensorflow 1.10.1 docker image

  - ```bash
    docker pull tensorflow/tensorflow:1.10.1-devel-py3
    export WORKSPACE=<path_to_workspace>
    docker run -it --rm -v $WORKSPACE:$WORKSPACE tensorflow/tensorflow:1.10.1-devel-py3
    ```

- run `export_inference_graph.py` to obtain model checkpoint ready for AIMET quantization

  ```bash
  cd models/research
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  python ./object_detection/export_inference_graph.py \
      --input_type image_tensor \
      --pipeline_config_path <path/to/modified_pipeline.config> \
      --trained_checkpoint_prefix <path/to/model.ckpt> \
      --output_directory <path/to/exported_model_directory> \
  ```

  - model checkpoint will be available at `<path/to/exported_model_directory>/AIMET/model.ckpt`

### COCO dataset TFRecord

TFRecord format of COCO dataset is need

- [download_and_preprocess_mscoco.sh](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh) can be used to download and convert coco dataset into TFRecord

  ```bash
  git clone https://github.com/tensorflow/models.git
  git checkout master
  cd models/research/object_detection/dataset_tools
  ./download_and_preprocess_mscoco.sh <mscoco_dir>
  ```

- If COCO dataset is already available or you want to download COCO dataset separately
  - COCO dataset can be download here: [COCO](https://cocodataset.org/#download)
    - Please download the 2017 Version
  - [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) can be used to convert dataset into TFRecord



## Additional Dependencies

|      Package      |
| :---------------: |
| tensorflow/models |
|    pycocotools    |

### Setup models Repo

- Clone the tensorflow models repository from github:

  ```bash
  git clone https://github.com/tensorflow/models.git
  cd models && git checkout r1.12.0 
  ```

- Append the repo location to your `PYTHONPATH` by doing the following:

  `export PYTHONPATH=<path to tensorflow models repo>/models/research:$PYTHONPATH`

### Pip install pycocotools

- Install pycocotools by running the following:  

  ```bash
  pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
  ```

### Adding dependencies within Docker Image

- If you are using a docker image, e.g. AIMET development docker, please add the following lines to the Dockerfile and rebuild the Docker image

```dockerfile
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```



## Usage
- `ssd_mobilenet_v2_quanteval.py` has four required arguments, an example usage is shown below
```bash
./ssd_mobilenet_v2_quanteval.py --model-checkpoint <path to model ckpt>/model.ckpt --dataset-dir <path to tfrecord dataset> --TFRecord-file-pattern 'coco_val.record-*-of-00010' --annotation-json-file <path to instances json file>/instances_val2017.json
```

- `--quantsim-output-dir` option can be used if want to save the quantized graph



## Quantizer Op Assumptions
In the evaluation script included, we have manually configured the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized