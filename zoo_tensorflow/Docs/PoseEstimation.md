# Tensorflow Pose Estimation

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further. This evaluation was run using [AIMET 1.22.2 for TensorFlow 1.15](https://github.com/quic/aimet/releases/tag/1.22.2) i.e. please set `release_tag="1.22.2"` and `AIMET_VARIANT="tf_gpu_tf115"` in the above instructions.

**NOTE:** This model is expected **not** to work with GPUs at or after NVIDIA 30-series (e.g. RTX 3050), as those bring a new architecture not fully compatible with TF 1.X.

## Additional Dependencies

|   Package   | Version |
| :---------: | :-----: |
| pycocotools |  2.0.2  |
|    scipy    |  1.1.0  |

### Adding dependencies within Docker Image

- If you are using a docker image, e.g. AIMET development docker, please add the following lines to the Dockerfile and rebuild the Docker image

```dockerfile
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip install scipy==1.1.0
```

## Model weights and configuration
- Downloading model checkpoints and configuration file for quantization is handled by evalution script.
- The pose estimation model can be downloaded from here:
  - [pose_estimation.tar.gz](/../../releases/download/pose_estimation/pose_estimation_tensorflow.tar.gz)
- This model has been compressed and its weights are optimized by applying DFQ (Data Free Quantization).

## Dataset 
- This evaluation script is built to evaluate on COCO2014 validation images with person keypoints. 
- coco dataset can be downloaded from here:
  - [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip)
  - [COCO 2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- The COCO dataset path should include coco images and annotations. It assumes a folder structure containing two subdirectories: images/val2014 and annotations. Corresponding images and annotations should be put into the two subdirectories.

## Usage
 - To run evaluation with QuantSim in AIMET, use the following 
  ```bash
  python pose_estimation_quanteval.py \
	--dataset_path < Path to COCO 2014 dataset> \
	--num-imgs < number of images to evaluate of COCO 2014 validation dataset> \
	--model-to-eval < which model to evaluate, two options are available: 'fp32' for evaluating original fp32 model, 'int8' for evaluating int8 quantized model >
  ```
- We only support evaluation on COCO 2014 val images with person keypoints.
- The results reported was evaluation on the whole dataset, which contains over 40k images and takes 15+ hours on a single RTX 2080Ti GPU. To run partial evaluation on MSCOCO validation dataset, specify --num-imgs argument.

## Quantization configuration 
- Weight quantization: 8 bits per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are  quantized
- 2K Images from COCO validation dataset are used as calibration dataset
