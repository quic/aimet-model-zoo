# Pytorch Pose Estimation

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.22/packaging/install.md) before proceeding further.
This model was tested with the `torch_gpu` variant of AIMET 1.22.2.

## Additional Dependencies

|   Package   | Version   |
| :---------: | :-----:   |
| pycocotools |  2.0.2    |
|    scipy    |  1.1.0    |

### Adding dependencies within Docker Image

- If you are using a docker image, e.g. AIMET development docker, please add the following lines to the Dockerfile and rebuild the Docker image

```dockerfile
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip install scipy==1.1.0
```
### Loading AIMET model zoo libraries 
```bash 
export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>
```

## Model checkpoints and configuration
- Downloading model checkpoints is handled by evaluation script pose_estimation_quanteval.py. Configuration is set to default by evaluation script pose_estimation_quanteval.py.
- The pose estimation model checkpoints can be downloaded here:
  - [Pose Estimation pytorch model](/../../releases/download/pose_estimation_pytorch/pose_estimation_pytorch_weights.tgz)
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json). (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset
- This evaluation script is built to evaluate on COCO2014 validation images with person keypoints. 
- COCO dataset can be downloaded from here:
  - [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip)
  - [COCO 2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- The COCO dataset path should include coco images and annotations. It assumes a folder structure containing two subdirectories: `images/val2014` and `annotations`. Corresponding images and annotations should be put into the two subdirectories.

## Usage
- To run evaluation with QuantSim in AIMET, use the following 
  ```bash
  python pose_estimation_quanteval.py \
	--dataset_path < Path to COCO 2014 dataset> \
	--use-cuda < Run evaluation using GPU > 
  ```

- The results reported was evaluation on the whole dataset, which contains over 40k images and takes ~5 hours on a single RTX 2080Ti GPU. So in case you want to run a faster evaluation, specifiy *num_imgs* argument to the second call with a small number to *evaluate_session* so that you run evaluation only on a partial dataset.

## Quantization configuration 
- Weight quantization: 8 bits per tensor asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are  quantized
- 2K Images from COCO validation dataset are used as calibration dataset
