# PyTorch-MobileNetV2

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

## Model modifications
1. Clone the [MobileNetV2 repo](https://github.com/tonylins/pytorch-mobilenet-v2)
```
git clone https://github.com/tonylins/pytorch-mobilenet-v2
cd pytorch-mobilenet-v2/
git checkout 99f213657e97de463c11c9e0eaca3bda598e8b3f
```
2. Place model definition under model directory
```
mkdir ../aimet-model-zoo/zoo_torch/examples/model
mv MobileNetV2.py ../aimet-model-zoo/zoo_torch/examples/model/
```
3. Download Optimized MobileNetV2 checkpoint from [Releases](/../../releases) and place under the model directory.
4. Replace all ReLU6 activations with ReLU
5. Following changes has been made or appended in original model definition for our suite 
  - Change line #87 as follows in MobileNetV2.py
```
self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
```
  - Change line #91 as follows in MobileNetV2.py
```
output_channel = int(c * width_mult)
```
  - Append line #100 as follows in MobileNetV2.py
```
self.features.append(nn.AvgPool2d(input_size // 32)
```
  - Change line #104 as follows in MobileNetV2.py
```
self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, n_class),
        )
```
  - Change line #110 as follows in MobileNetV2.py
```
x = x.squeeze()
```
## Obtaining model checkpoint and dataset

- The original MobileNetV2 checkpoint can be downloaded here:
  - https://github.com/tonylins/pytorch-mobilenet-v2
- Optimized MobileNetV2 checkpoint can be downloaded from the [Releases](/../../releases) page.
- ImageNet can be downloaded here:
  - http://www.image-net.org/

## Usage
- To run evaluation with QuantSim in AIMET, use the following
```bash
python eval_mobilenetv2.py \
	--model-path <path to optimized mobilenetv2 checkpoint> \
	--images-dir <path to imagenet root directory> \
	--quant-scheme <quantization schme to run> \
	--input-shape <input shape to model> \
	--default-output-bw <bitwidth for activation quantization> \
	--default-param-bw <bitwidth for weight quantization>     		
```

## Quantization Configuration
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
- TF_enhanced was used as quantization scheme
- Data Free Quantization and Quantization aware Training has been performed on the optimized checkpoint
