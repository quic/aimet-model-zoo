# SRGAN (Super Resolution)

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Additional Dependencies

|   Package    | Version |
| :----------: | :-----: |
| scikit-image | 0.16.2  |
|     mmcv     |  1.2.0  |

### Setup Super-resolution repo

- Clone the <a href="https://github.com/krasserm/super-resolution">krasserm/super-resolution</a> repo

  `git clone https://github.com/krasserm/super-resolution.git`

- Append the repo location to your `PYTHONPATH` with the following:

  `export PYTHONPATH=<path to super-resolution repo>/super-resolution:$PYTHONPATH`



### Adding dependencies within Docker Image

- If you are using a docker image, e.g. AIMET development docker, please add the following lines to the Dockerfile and rebuild the Docker image

```dockerfile
RUN pip install scikit-image==0.16.2
RUN pip install mmcv==1.2.0
```



## Obtaining model weights and dataset

- The SRGAN model can be downloaded from:
  - <a href="https://github.com/krasserm/super-resolution">krasserm/super-resolution</a>
- Three benchmark dataset can be downloaded here:
  - [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
  - [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
  - [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
- If you want to use custom high resolution images, one way to generate corresponding low resolution images can be found at <a href="https://github.com/krasserm/super-resolution/issues/19">this issue</a>
  - with a Python version of MATLAB `imresize` function available <a href="https://github.com/fatheral/matlab_imresize">here</a>



## Usage

- The `srgan_quanteval.py` script requires two arguments to run: weights_path, images_path.
  These are positional arguments so you just have to specify the arguments in order.

    ```bash
  python ./zoo_tensorflow/examples/srgan_quanteval.py [--options] <path to model file> <path to dataset>
    ```

- we only support 4x super resolution on .png images. So make sure you high resolution images are 4x the dimension of you low resolution images. If you are using one of the benchmark dataset, please use images under `image_SRF_4` directory

- We assume low and high resolution images are both present under the same directory,

  with images following naming conventions:

  - low resolution images will have file name suffix: `LR.png`
    - e.g. `people_LR.png`
  - high resolution images will have file name suffix: `HR.png`
    - e.g. `people_HR.png`
    
    
## Quantizer Op Assumptions
In the evaluation script included, we have modified activation bitwidth, the configuration looks like below:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 16 bits, asymmetric quantization
- Model inputs are not quantized
- Bias Correction and Cross Layer Equalization have been applied
