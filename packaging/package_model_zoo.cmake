# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# CMake file to generate AIMET model zoo packages

set(src_packaging_dir "${SOURCE_DIR}/packaging")
set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")

# First delete the existing packaging directory if it exists
file(REMOVE_RECURSE ${build_packaging_dir})

# Setup Tensorflow package dependencies if required
if(ENABLE_TENSORFLOW)
    # Add AIMET Tensorflow package to package array list
    list(APPEND package_name_list aimet_zoo_tensorflow)
endif()

# Setup Torch package dependencies if required
if(ENABLE_TORCH)
    # Add AIMET Torch package to package array list
    list(APPEND package_name_list aimet_zoo_torch)
endif()

# Set the deployment paths
set(ZOO_PACKAGE_NAME ${PROJECT_NAME}-${SW_VERSION})
set(ZOO_PACKAGE_DIR ${build_packaging_dir}/${ZOO_PACKAGE_NAME})
file(MAKE_DIRECTORY ${ZOO_PACKAGE_DIR})

# Loop over the package array list to generate wheel files
foreach(package ${package_name_list})

    # Rename the package TOML file to the expected name "pyproject.toml"
    configure_file("${src_packaging_dir}/${package}_pyproject.toml" "${build_packaging_dir}/pyproject.toml" COPYONLY)

    # Populate the packaging directory with the python code files
    file(COPY ${AIMET_PACKAGE_PATH}/${package} DESTINATION ${build_packaging_dir}/)

    # Location of the package folder
    set(package_dir "${build_packaging_dir}/${package}")
    # Location of the subfolder within package
    set(package_bin_dir "${package_dir}/bin")

    # Copy over additional essential files into the wheel package
    configure_file("${SOURCE_DIR}/LICENSE.pdf" "${package_bin_dir}/" COPYONLY)
    configure_file("${SOURCE_DIR}/NOTICE.txt" "${package_bin_dir}/" COPYONLY)

    #TODO We need to symlink the top-level .git into the packaging directory since package 
    # generation using TOML does not seem to work by default in out-of-source build paths.
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${SOURCE_DIR}/.git" "${build_packaging_dir}/.git")

    # Invoke the command to create the wheel packages.
    execute_process(COMMAND python3 -m pip wheel . WORKING_DIRECTORY ${build_packaging_dir} OUTPUT_VARIABLE output_var)

    # Now delete all the intermediate artifacts
    file(REMOVE "${build_packaging_dir}/.git")
    file(REMOVE_RECURSE "${build_packaging_dir}/${package}")
    file(REMOVE_RECURSE "${build_packaging_dir}/build")
    file(REMOVE "${build_packaging_dir}/pyproject.toml")

    file(GLOB WHEEL_FILE "${build_packaging_dir}/*.whl")

    file(COPY ${WHEEL_FILE} DESTINATION ${ZOO_PACKAGE_DIR})

endforeach()

# Copy over additional essential files into the package folder
configure_file("${SOURCE_DIR}/LICENSE.pdf" ${ZOO_PACKAGE_DIR} COPYONLY)
configure_file("${SOURCE_DIR}/NOTICE.txt" ${ZOO_PACKAGE_DIR} COPYONLY)

# Create a tarball archive of the entire model zoo package
execute_process(COMMAND
    ${CMAKE_COMMAND} -E chdir ${build_packaging_dir}
    ${CMAKE_COMMAND} -E tar cvzf ${build_packaging_dir}/${ZOO_PACKAGE_NAME}.tar.gz ${ZOO_PACKAGE_DIR})
