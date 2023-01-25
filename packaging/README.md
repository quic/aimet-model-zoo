# AIMET model zoo package
This page provides information on the installable python pip packages for the AIMET model zoo and the instructions to generate them.

## Table of Contents
- [Install package](#install-package)
- [Build new package](#build-new-package)

## Install package
> _NOTE:_ Follow this section in order to install and use a released version of the AIMET model zoo package (*recommended*).

Go to https://github.com/quic/aimet-model-zoo/releases and identify the release tag of the package you want to install. 

Set the `<variant_string>` to ONE of the following depending on your desired variant
- For the PyTorch variant, use `"torch"`
- For the TensorFlow variant, use `"tensorflow"`
```bash
export aimet_zoo_variant=<variant_string>
```

Replace `<release_version>` in the steps below with the appropriate tag:
```bash
export release_version=<release_version>
```

Set the package download URL as follows:
```bash
export download_url="https://github.com/quic/aimet-model-zoo/releases/download/${release_version}"
```

Install the AIMET packages in the order specified below:
```bash
python3 -m pip install ${download_url}/aimet_zoo_${aimet_zoo_variant}-${release_version}-py3-none-any.whl
```

> _NOTE:_ Dependencies (if any) will NOT get installed automatically. For any additional pre-requisite packages, please check the appropriate *<model>.md* within the <model> subfolder in [TensorFlow](aimet_zoo_tensorflow) or [PyTorch](aimet_zoo_torch) folders corresponding to your model(s) of interest.

## Build new package
> _NOTE:_ Follow this section in order to locally build a new AIMET model zoo package(s) from sources.

Please read [this section](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#requirements) for the host pre-requisities and [this section](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#setup-the-environment) for the recommended environment setup.

Upgrade the setuptools package to the latest version as follows:  
`python3 -m pip install --upgrade setuptools`

To obtain the code, first define a workspace and follow these instructions:

```bash
WORKSPACE="<absolute_path_to_workspace>"
mkdir $WORKSPACE && cd $_
git clone https://github.com/quic/aimet-model-zoo.git
```

Follow these instructions to build the AIMET model zoo code:

> NOTE: If you are inside the docker, set `WORKSPACE="<absolute_path_to_workspace>"` again.
```bash
cd $WORKSPACE/aimet
mkdir build && cd build

# Run cmake (set flags in the below command depending on your needs)
# To include torch, use -DENABLE_TORCH=ON. To exclude torch, use -DENABLE_TORCH=OFF.
# To include tensorflow, use -DENABLE_TENSORFLOW=ON. To exclude tensorflow, use -DENABLE_TENSORFLOW=OFF.
cmake .. -DENABLE_TORCH=ON -DENABLE_TENSORFLOW=ON -DDEPLOYMENT_PATH="<path_to_deploy_tarball>" -DPIP_CONFIG_FILE="<pip_repo_config_file>" -DPIP_INDEX="<name_of_repo_index>" -DPIP_CERT_FILE="<ca_certificate_file>"

make

make install
```

Generate the package as follows:  
`make packagemodelzoo`

Once the installation step is complete, the wheel files AND the consolidated tarball package is created at: `$WORKSPACE/aimet-model-zoo/build/packaging/`.

Deploy the package tarball to the location specified (via the cmake command above):  
`make deploy`

Upload the python pip package to your repository (using the destination URL specified via the cmake command above):  
`make upload`
> NOTE: The upload make target does NOT work yet. You may need to upload manually until this is fixed.

Install the package(s) that you just built as follows:  
```bash
python3 -m pip install $WORKSPACE/aimet-model-zoo/build/packaging/aimet_zoo_${aimet_zoo_variant}-${release_version}-py3-none-any.whl
```
