# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# CMake file to upload the AIMET model zoo wheel packages

set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")
file(GLOB wheel_files "${build_packaging_dir}/*.whl")

# Check the pip config file and set it to a default value (if not set)
if(PIP_CONFIG_FILE STREQUAL "None")
    set(PIP_CONFIG_FILE "~/.pypirc")
    message(WARNING "PIP_CONFIG_FILE was not specified. Setting it to ${PIP_CONFIG_FILE}.")
else()
    message(STATUS "PIP_CONFIG_FILE already set to ${PIP_CONFIG_FILE}.")
endif()

# Check whether the pip index was specified (must be present within the pip config file)
if(PIP_INDEX STREQUAL "None")
    message(FATAL_ERROR "PIP_INDEX was not set. Please cmake -DPIP_INDEX=<pip_index_value>.")
endif()

# Set the pip package upload command argument string
set(pip_upload_cmd_args " upload --verbose -r ${PIP_INDEX} --config-file ${PIP_CONFIG_FILE} ")

# Check the certificate path and append to the command root string if present
if(PIP_CERT_FILE STREQUAL "None")
    message(WARNING "PIP_CERT_FILE was not specified. Not using that option with twine command.")
else()
    set(pip_upload_cmd_args "${pip_upload_cmd_args} --cert ${PIP_CERT_FILE}")
endif()

# Loop over the package array list to select the wheel files to be uploaded
foreach(wheel_file ${wheel_files})
    # Pre-pend the twine command and add the wheel file to be uploaded at the end
    set(pip_upload_cmd twine "${pip_upload_cmd_args} ${wheel_file}")
    message(STATUS "Package upload command: ${pip_upload_cmd}")

    # execute the command to upload the wheel files.
    execute_process(COMMAND ${pip_upload_cmd} WORKING_DIRECTORY ${build_packaging_dir} OUTPUT_VARIABLE output_var ERROR_VARIABLE error_var RESULT_VARIABLE result_var)
    if(result_var EQUAL "1")
        message( FATAL_ERROR "twine upload failed")
    endif()

    message(WARNING "Package upload MAY not have completed. Please check destination and upload manually if needed.")
endforeach()
