# SRGAN (Super Resolution)

## Setup AI Model Efficiency Toolkit (AIMET)
Please [install and setup AIMET](../../README.md#install-aimet) before proceeding further.

### Setup Super-resolution repo

- Clone the <a href="https://github.com/andreas128/mmsr">mmsr</a> repo  
  `git clone https://github.com/andreas128/mmsr.git`  
  `git checkout a73b318f0f07feb6505ef5cb1abf0db33e33807a`

- Comment out the following line 10 in mmsr/codes/models/archs/EDVR_arch.py

  ```raise ImportError('Failed to import DCNv2 module.')```

- Append the repo location to your `PYTHONPATH` with the following:  
  `export PYTHONPATH=<path to mmsr repo>:<path to mmsr repo>/codes:$PYTHONPATH`
  
  Note that here we add both mmsr and the subdirectory mmsr/codes to our path.
  
 - Find mmsr/codes/models/archs/arch_util.py and do the following changes:
   1. In \_\_init__ append one line ```self.relu=nn.ReLU()``` after 
   ```self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)``` like below:
   
   ```python
    super(ResidualBlock_noBN, self).__init__()
    self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    self.relu = nn.ReLU()
   ```
   
   2. In forward replace ```out = F.relu(self.conv1(x), inplace=True)```
    with ```out = self.relu(self.conv1(x))``` like below:
    
   ```python
    identity = x
    # out = F.relu(self.conv1(x), inplace=True)
    out = self.relu(self.conv1(x))
    out = self.conv2(out)
   ```
   
   These changes are necessary since AIMET currently doesn't run on some pytorch 
   functionals.
   
## Obtaining model weights and dataset

- The SRGAN model can be downloaded from:
  - <a href="/../../releases/tag/srgan_mmsr_model">mmediting</a>
    
- Three benchmark dataset can be downloaded here:
  - [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
  - [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
  - [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
  
  Our benchmark results use images under **image_SRF_4** directory which tests 4x
  super-resolution as the suffix number indicates. You can also use other scales.
  See instructions for usage below.

## Usage

- The `srgan_quanteval.py` script requires you to specify a .yml file which contains locations to your dataset and .pth model together with some config parameters. You can just pass the mmsr/codes/options/test/test_SRGAN.yml as your .yml file. Remember to edit the file s.t.
  - dataroot_GT points to your directory of HR images
  - dataroot_LQ points to your directory of LR images
  - pretrain_model_G points to where you store your srgan .pth file
  - scale has to match the super-resolution images' scale
  
Run the script as follows:
  ```bash
  python ./zoo_torch/examples/srgan_quanteval.py [--options] -opt <path to .yml file>
  ```


## Quantizer Op Assumptions
In the evaluation script included, we have used the default config file, which configures the quantizer ops with the following assumptions:
- Weight quantization: 8 bits, asymmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are not quantized
