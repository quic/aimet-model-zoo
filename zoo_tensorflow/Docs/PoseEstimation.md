# Pose Estimation

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

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

## Obtaining model weights and dataset

- The pose estimation model can be downloaded here:
  - <a href="/../../releases/download/pose_estimation/pose_estimation_tensorflow.tar.gz">
    pose_estimation.tar.gz
    </a>
- This model has been compressed and its weights are optimized by applying DFQ 
    (Data Free Quantization).

- coco dataset can be downloaded here:
  - <a href="http://images.cocodataset.org/zips/val2014.zip">COCO 2014 Val images</a>
  - <a href="http://images.cocodataset.org/annotations/annotations_trainval2014.zip">
    COCO 2014 Train/Val annotations
    </a>


## Usage

- The program requires two arguments to run: model_meta_file_dir, coco_path. These are positional 
  arguments so you must specify the arguments in order.
  
  ```bash
  python ./examples/pose_estimation_quanteval.py <path to model meta file> <path to location of coco dataset>
  ```
  
- We only support evaluation on COCO 2014 val images with person keypoints.
  
- The results reported was evaluation on the whole dataset, which contains over 40k 
  images and takes 15+ hours on a single RTX 2080Ti GPU. So in case you want to run 
  a faster evaluation, specifiy <em>num_imgs</em> argument to the second call with a 
  small number to  <em>evaluate_session</em> so that you run evaluation only on a 
  partial dataset. 