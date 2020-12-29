# RetinaNet

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies

|     Package     |
| :-------------: |
| keras-retinanet |
|   pycocotools   |

### Setup RetinaNet Repo

- Clone the RetinaNet repository from github: https://github.com/fizyr/keras-retinanet

  ```git clone https://github.com/fizyr/keras-retinanet.git  ```

  Within the cloned repository, checkout the commit corresponding to pre-tf2.0.  The included example scripts only works for TF 1.x.

  ```git checkout 08af308d01a8f22dc286d62bc26c8496e1ff6539```

  Install keras-retinanet and dependencies using by running,

  ```pip install . --user```

### Pip install pycocotools

- Install pycocotools by running the following:  
    ```bash
    pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
    ```
### Adding dependencies within Docker Image

- If you are using a docker image, e.g. AIMET development docker, please add the following lines to the Dockerfile and rebuild the Docker image

```dockerfile
RUN git clone https://github.com/fizyr/keras-retinanet.git /tmp/keras-retinanet/
RUN cd /tmp/keras-retinanet/ && git checkout 08af308d01a8f22dc286d62bc26c8496e1ff6539
RUN cd /tmp/keras-retinanet/ && pip install .
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```



## Obtaining model weights and dataset

- The original pre-trained keras retinanet model can be downloaded here:
  - <a href="https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5">RetinaNet pre-trained model</a>
- coco dataset can be downloaded here:
  - http://cocodataset.org



## Usage
- The example script requires paths to coco dataset and keras retinanet model (look at the above *Obtaining model weights and dataset* instructions to download).
- There are two actions ```retinanet_quanteval.py``` can perform, ```eval_original``` will evaluate the accuracy of the original model, while ```eval_quantized``` will quantize the original model and evaluate the accuracy on the quantized model
```
python ./examples/retinanet_quanteval.py coco <path to location of coco dataset> <path to location of original retinanet model> --action eval_original

python ./examples/retinanet_quanteval.py coco <path to location of coco dataset> <path to location of original retinanet model> --action eval_quantized
```
