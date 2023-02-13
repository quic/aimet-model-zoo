#!/bin/bash
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at 
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# ----------------------------------------------
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# ----------------------------------------------

###############################################################################
## This is a script to build and run tests on AIMET model zoo folder.
## This script must be run outside the docker container from within the AIMET model zoo top-level folder.
## For help and usage information: buildntest.sh -h
###############################################################################

# enable exit on error.
set -e

workspaceFolder=`pwd`
workspaceFolder=`readlink -f ${workspaceFolder}`
outputRootFolder=$workspaceFolder
scriptPath=`readlink -f $(dirname "$0")`
entrypoint="${scriptPath}/dobuildntest.sh"
options_string=""
EXIT_CODE=0

interactive_mode=0
dry_run=0
loading_symbol="..."

usage() {
  echo -e "\nThis is a script to build and run tests on AIMET code."
  echo -e "This script must be executed from within the AIMET repo's top-level folder."
  echo -e "NOTE: This script will build and start a docker container.\n"
  
  echo "${0} [-o <output_folder>]"
  echo "    -b --> build the code"
  echo "    -p --> generate pip packages"
  echo "    -v --> run code violation checks (using pylint tool)"
  echo "    -o --> optional output folder. Default is current directory"
  echo "    -i --> just build and start the docker in interactive mode (shell prompt)"
  echo "    -m --> mount the volumes (For multiple environment variables, use the same option"
  echo "            multiple times ex. -m <path1> -m <path2> ...)"
  echo "    -e --> set existing env var from current environment (For multiple environment "
  echo "            variables, use the same option multiple times ex. -e <var1> -e <var2> ...)"
  echo "    -y --> set any custom script/command as entrypoint for docker (default is dobuildntest.sh)"
  echo "    -n --> dry run mode (just display the docker command)"
  echo "    -l --> skip docker image build and pull from code linaro instead"
}


while getopts "o:bce:ilm:nphvy:" opt;
   do
      case $opt in
         b)
             options_string+=" -b"
             ;;
         e)
             USER_ENV_VARS+=("$OPTARG")
             ;;
         p)
             options_string+=" -p"
             ;;
         v)
             options_string+=" -v"
             ;;
         h)
             usage
             exit 0
             ;;
         i)
             interactive_mode=1
             ;;
         m)
             USER_MOUNT_DIRS+=("$OPTARG")
             ;;
         l)
             USE_LINARO=1
             ;;
         n)
             dry_run=1
             loading_symbol=":"
             ;;
         o)
             outputRootFolder=$OPTARG
             ;;
         y)
             entrypoint=$OPTARG
             ;;
         :)
             echo "Option -$OPTARG requires an argument" >&2
             exit 1;;
         ?)
             echo "Unknown arg $opt"
             usage
             exit 1
             ;;
      esac
done

if [ ${dry_run} -eq 0 ]; then
	set -x
fi

timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Set the docker file path
# This is the default value (for python 3.8 docker files)
dockerfile_path=${scriptPath}

# Set the root URL that hosts the pre-built development docker image
if [ -n "$AIMET_PREBUILT_DOCKER_IMAGE_URL" ]; then
    # Use a custom source if one is provided
    prebuilt_docker_image_url=${AIMET_PREBUILT_DOCKER_IMAGE_URL}
else
    # Use the default docker images from Code Linaro
    prebuilt_docker_image_url="artifacts.codelinaro.org/codelinaro-aimet"
fi

# Select the docker file based on the build variant
if [ -n "$AIMET_ZOO_VARIANT" ]; then
    docker_file="Dockerfile.${AIMET_ZOO_VARIANT}"
    docker_image_name="aimet-dev-docker:${AIMET_ZOO_VARIANT}"
    prebuilt_docker_image_name="${prebuilt_docker_image_url}/aimet-dev:latest.${AIMET_ZOO_VARIANT}"
else
    docker_file="Dockerfile"
    docker_image_name="aimet-dev-docker:latest"
    prebuilt_docker_image_name="${prebuilt_docker_image_url}/aimet:latest"
fi

# Either use code linaro docker image or build it from scratch
if [ -n "$USE_LINARO" ]; then
    docker_image_name=$prebuilt_docker_image_name
    echo -e "*** Using pre-built docker image: $docker_image_name ***\n"
else
    echo -e "*** Building docker image${loading_symbol} ***\n"
    pushd ${dockerfile_path}
    DOCKER_BUILD_CMD="docker build -t ${docker_image_name} -f ${docker_file} ."
    if [ $interactive_mode -eq 1 ] && [ $dry_run -eq 1 ]; then
        echo ${DOCKER_BUILD_CMD}
        echo
    else
        eval ${DOCKER_BUILD_CMD}
    fi
    popd
fi

if [[ -z "${BUILD_NUMBER}" ]]; then
    # If invoked from command line by user, use a timestamp suffix
    results_path=${outputRootFolder}/buildntest_results/$timestamp
    docker_container_name=aimet-dev_${USER}_${timestamp}
    # If this is a variant, then append the variant string as suffix
    if [ -n "$AIMET_ZOO_VARIANT" ]; then
        docker_container_name="${docker_container_name}_${AIMET_ZOO_VARIANT}"
        results_path=${results_path}_${AIMET_ZOO_VARIANT}
    fi
else
    # If invoked from jenkins, add username, build num, and timestamp
    results_path=${outputRootFolder}/buildntest_results
    docker_container_name=aimet-dev_${USER}_${BUILD_NUMBER}_${timestamp}
fi

# Add desired output folder to the options string
options_string+=" -o ${results_path}"

rm -rf {results_path} | true
mkdir -p ${results_path}

# Kill any previous running containers by the same name
if [[ -z "${BUILD_NUMBER}" ]]; then
    # If invoked from command line by user, kill any previous running containers by the same name
    docker ps | grep ${docker_container_name} && docker kill ${docker_container_name} || true
else
    # If invoked from jenkins, kill anly previous running containers starting with "aimet-dev_"
    containers=($(docker ps | awk '/aimet-dev_/ {print $NF}'))
    for container_name in "${containers[@]}"; do 
        docker kill "$container_name" || true;
    done
fi

# Add data dependency path as additional volume mount if it exists
if [ -n "${DEPENDENCY_DATA_PATH}" ]; then
   docker_add_vol_mount+=${DEPENDENCY_DATA_PATH}
   USER_ENV_VARS+=("DEPENDENCY_DATA_PATH")
else
   # If it does not exist, then just add the path of the current script since we cannot leave it 
   # empty
   docker_add_vol_mount+=${scriptPath}
fi

# Check if and which version of nvidia docker is present
set +e
DOCKER_RUN_PREFIX="docker run"
dpkg -s nvidia-container-toolkit > /dev/null 2>&1
NVIDIA_CONTAINER_TOOKIT_RC=$?
dpkg -s nvidia-docker > /dev/null 2>&1
NVIDIA_DOCKER_RC=$?
dpkg -s nvidia-docker2 > /dev/null 2>&1
NVIDIA_DOCKER_RC2=$?
set -e

if [ -n "$AIMET_ZOO_VARIANT" ] && [[ "$AIMET_ZOO_VARIANT" == *"cpu"* ]]; then
    echo "Running docker in CPU mode..."
    DOCKER_RUN_PREFIX="docker run"
elif [ $NVIDIA_DOCKER_RC -eq 0 ] || [ $NVIDIA_DOCKER_RC2 -eq 0 ]; then
    echo "Running docker in GPU mode using nvidia-docker..."
    DOCKER_RUN_PREFIX="nvidia-docker run"
elif [ $NVIDIA_CONTAINER_TOOKIT_RC -eq 0 ]; then
    echo "Running docker in GPU mode using nvidia-container-toolkit..."
    DOCKER_RUN_PREFIX="docker run --gpus all"
else
    echo "ERROR: You requested GPU mode, but no nvidia support was detected!"
    exit 3
fi

echo -e "Starting docker container${loading_symbol} \n"
DOCKER_RUN_CMD="${DOCKER_RUN_PREFIX} --rm --name=$docker_container_name -e DISPLAY=:0 \
				-u $(id -u ${USER}):$(id -g ${USER}) \
				-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
				-v /tmp/.X11-unix:/tmp/.X11-unix \
				-v ${workspaceFolder}:${workspaceFolder} \
				-v ${outputRootFolder}:${outputRootFolder} \
				-v ${docker_add_vol_mount}:${docker_add_vol_mount} \
				-v /etc/localtime:/etc/localtime:ro \
				-v /etc/timezone:/etc/timezone:ro --network=host --ulimit core=-1 \
				-w ${workspaceFolder} \
				--ipc=host --shm-size=8G"

# Check if HOME variable is set
if [[ -v HOME ]]; then
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v ${HOME}:${HOME}"
fi

# Set env variables if requested
for user_env_var in "${USER_ENV_VARS[@]}"; do
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -e ${user_env_var}=\$${user_env_var}"
done

# Mount directories if requested
for user_dir in "${USER_MOUNT_DIRS[@]}"; do
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v ${user_dir}:${user_dir}"
done

if [ $interactive_mode -eq 1 ]; then
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -it --hostname aimet-dev ${docker_image_name}"
	if [ $dry_run -eq 1 ]; then
		echo ${DOCKER_RUN_CMD}
		echo
	else
		eval ${DOCKER_RUN_CMD}
	fi
else
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} --entrypoint=${entrypoint} \
	${docker_image_name} ${options_string} -w ${workspaceFolder} \
	-o ${results_path} | tee ${results_path}/full_log.txt"
	eval ${DOCKER_RUN_CMD}

	# Capture the status of the docker command prior to the tee pipe
	EXIT_CODE=${PIPESTATUS[0]}

	if [ ${EXIT_CODE} -ne 0 ]; then
	    echo -e "Docker execution of stage failed!"
	elif [ ! -f "${results_path}/summary.txt" ]; then
	    echo -e "Failed to launch any build or test stages!"
	    EXIT_CODE=3
	elif grep -q FAIL "${results_path}/summary.txt"; then
	    echo -e "One or more stages failed!"
	    EXIT_CODE=3
	fi

	exit $EXIT_CODE
fi
