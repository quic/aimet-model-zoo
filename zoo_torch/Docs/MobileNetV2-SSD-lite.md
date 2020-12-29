# PyTorch-MobileNetV2-SSD-lite

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Model modifications
1. Clone the original repository
```
git clone https://github.com/qfgaohao/pytorch-ssd.git
cd pytorch-ssd
git checkout f61ab424d09bf3d4bb3925693579ac0a92541b0d
git apply ../aimet-model-zoo/zoo_torch/examples/torch_ssd_eval.patch
```
2. Place the model definition & eval_ssd.py to aimet-model-zoo/zoo_torch/examples/
```
mv vision ../aimet-model-zoo/zoo_torch/examples/
mv eval_ssd.py ../aimet-model-zoo/zoo_torch/examples/
```
3. Change __init__ function from line #27 in vision/ssd/ssd.py as follows:
```
self.config = None #############Change 1

self.image_size = 300
self.image_mean = np.array([127, 127, 127])  # RGB layout
self.image_std = 128.0
self.iou_threshold = 0.45
self.center_variance = 0.1
self.size_variance = 0.2

self.specs = [box_utils.SSDSpec(19, 16, box_utils.SSDBoxSizes(60, 105), [2, 3]),
              box_utils.SSDSpec(10, 32, box_utils.SSDBoxSizes(105, 150), [2, 3]),
              box_utils.SSDSpec(5, 64, box_utils.SSDBoxSizes(150, 195), [2, 3]),
              box_utils.SSDSpec(3, 100, box_utils.SSDBoxSizes(195, 240), [2, 3]),
              box_utils.SSDSpec(2, 150, box_utils.SSDBoxSizes(240, 285), [2, 3]),
              box_utils.SSDSpec(1, 300, box_utils.SSDBoxSizes(285, 330), [2, 3])]

self.gen_priors = box_utils.generate_ssd_priors(self.specs, self.image_size)

# register layers in source_layer_indexes by adding them to a module list
self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                           if isinstance(t, tuple) and not isinstance(t, GraphPath)])

if device:
    self.device = device
else:
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if is_test:
    self.priors = self.gen_priors.to(self.device)
```
4. Change line #93 in vision/ssd/ssd.py as follows:
```
boxes = box_utils.convert_locations_to_boxes(
                locations.cpu(), self.priors.cpu(), self.center_variance, self.size_variance
)
```

## Obtaining model checkpoint and dataset
- The original MobileNetV2-SSD-lite checkpoint can be downloaded here:
  - https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
- Optimized checkpoint can be downloaded from the [Releases](/../../releases).
- Pascal VOC2007 dataset can be downloaded here:
  - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_ssd.py \
 --net <Architecture to run, currently only 'mb2-ssd-lite' is supported> \
 --trained_model <Path to checkpoint to load> \
 --dataset <The root directory of dataset> \
 --label_file <Path to label file to parse> \
 --eval_dir <Path to save a result>
```

## Quantization Configuration
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Cross-layer-Equalization and Adaround have been applied on optimized checkpoint
