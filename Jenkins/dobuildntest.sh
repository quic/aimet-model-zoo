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
## This is a script to build and run tests on AIMET model zoo code.
## This script must be run within the docker container from the AIMET model zoo top-level folder
###############################################################################

# verbose mode
# set -x

# enable exit on error.
set -e

run_prep=1
run_clean=0
run_build=0
run_package_gen=0
run_code_violation=0

EXIT_CODE=0

workspaceFolder=`pwd`
outputFolder=

function pre_exit {
    # Capture the exit code
    EXIT_CODE=$?

    if [ -z "$outputFolder" ]; then
        outputFolder=$workspaceFolder/buildntest_results
    fi

    summaryFile=${outputFolder}/summary.txt

    if [[ -f ${summaryFile} ]]; then
        # In case there is non-zero exit code, then add a FAILED tag to the summary file.
        if [ $EXIT_CODE -ne 0 ]; then
            echo -e "One or more Stages \t\t FAILED " | tee -a ${outputFolder}/summary.txt
        fi

        echo -e "----------------------------------------------------------------------------------------------------------\n" |tee -a ${summaryFile}
        echo -e "\nResults are in location:\n${outputFolder}\n" | tee -a ${summaryFile}
        cat ${summaryFile}

        if grep -q FAIL "${summaryFile}"; then
            EXIT_CODE=3
        fi
    fi

    # Return the exit code
    exit ${EXIT_CODE}
}
trap pre_exit EXIT

function check_stage() {
    RESULT=$1
    STAGE=$2

    if [ "$3" ]; then
        EXIT_ON_FAIL=$3
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "Stage $STAGE \t\t PASS " | tee -a ${outputFolder}/summary.txt
    else
        echo -e "Stage $STAGE \t\t FAILED " | tee -a ${outputFolder}/summary.txt
        if [ $EXIT_ON_FAIL == "true" ]; then
            echo -e "\n ABORTED " | tee -a ${outputFolder}/summary.txt
            exit 3
        fi
    fi
}

usage() {
  echo -e "\nThis is a script to build and run tests on AIMET code."
  echo -e "This script must be executed from within the AIMET repo's top-level folder."
  echo -e "NOTE: This script must be executed within the docker container (or in a machine with all dependencies installed). It will NOT start a docker container.\n"
  echo "${0} [-o <output_folder>]"
  echo "    -b --> build the code"
  echo "    -p --> generate pip packages"
  echo "    -v --> run code violation checks (using pylint tool)"
}

while getopts "o:w:bchpv" opt;
   do
      case $opt in
         b)
             run_build=1
             ;;
         c)
             run_clean=1
             ;;
         p)
             run_package_gen=1
             ;;
         v)
             run_code_violation=1
             ;;
         h)
             usage
             exit 0
             ;;
         o)
             outputFolder=$OPTARG
             ;;
         w)
             workspaceFolder=$OPTARG
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

if [[ -z "${workspaceFolder}" ]]; then
    usage
    echo -e "ERROR: Workspace directory was not specified!"
    exit 3
fi

echo "Starting AIMET build and test..."
workspaceFolder=`readlink -f ${workspaceFolder}`
buildFolder=$workspaceFolder/build

if [[ -d "${workspaceFolder}/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/Jenkins
elif [[ -d "${workspaceFolder}/aimet/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/aimet/Jenkins
fi

if [ $run_clean -eq 1 ]; then
    echo -e "\n********** Stage: Clean **********\n"
    if [ -d ${buildFolder} ]; then
        rm -rf ${buildFolder}/* | true
    fi
fi

if [ -z "$outputFolder" ]; then
    outputFolder=$buildFolder/results
fi
mkdir -p ${outputFolder}
if [ ! -f "${outputFolder}/summary.txt" ]; then
    touch ${outputFolder}/summary.txt
fi
if ! grep -q "AIMET Build and Test Summary" "${outputFolder}/summary.txt"; then
    echo -e "\n----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
    echo -e "\t\t AIMET Build and Test Summary " | tee -a ${outputFolder}/summary.txt
    echo -e "----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
fi

if [ $run_prep -eq 1 ]; then
    echo -e "\n********** Stage 1: Preparation **********\n"
    cd $workspaceFolder
fi

if [ $run_build -eq 1 ]; then
    echo -e "\n********** Stage 2: Build **********\n"
    echo -e $buildFolder
    mkdir -p $buildFolder
    cd $buildFolder

    extra_opts=""

    # Add build options based on variant
    if [ -n "$AIMET_ZOO_VARIANT" ]; then
        if [[ "$AIMET_ZOO_VARIANT" == *"tf"* ]]; then
            extra_opts+=" -DENABLE_TENSORFLOW=ON"
        fi
        if [[ "$AIMET_ZOO_VARIANT" == *"torch"* ]]; then
            extra_opts+=" -DENABLE_TORCH=ON"
        fi
        if [[ "$AIMET_ZOO_VARIANT" != *"tf"* ]]; then
            extra_opts+=" -DENABLE_TENSORFLOW=OFF"
        fi
        if [[ "$AIMET_ZOO_VARIANT" != *"torch"* ]]; then
            extra_opts+=" -DENABLE_TORCH=OFF"
        fi
    fi
    # Do not exit on failure by default from this point forward
    set +e
    echo -e ${pwd}
    echo -e "Make stage"
    echo -e ${extra_opts}
    cmake ${extra_opts} ..

    make -j 8
    check_stage $? "Build" "true"

fi

if [ $run_package_gen -eq 1 ]; then
    cd $buildFolder

    echo -e "\n********** Stage 2b: Install **********\n"
    make install
    check_stage $? "Install" "true"

    echo -e "\n********** Stage 2c: Package **********\n"
    make packagemodelzoo
    check_stage $? "Package" "true"
fi

if [ $run_code_violation -eq 1 ]; then
    cd $buildFolder

    echo -e "\n********** Stage 4: Code violation checks **********\n"
    make pylintmodelzoo
    check_stage $? "Code Violations" "true"
fi

exit $EXIT_CODE

