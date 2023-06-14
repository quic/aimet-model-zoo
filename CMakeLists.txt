# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

cmake_minimum_required(VERSION 3.17)

project(aimet-model-zoo)

# -------------------------------
# Conditional build for CUDA
# -------------------------------
if (NOT (DEFINED ENABLE_CUDA))
  message("Compiling with CUDA not explicitly disabled. Enabling implicitly")
  set(ENABLE_CUDA ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_CUDA))

message("Cuda: " ${ENABLE_CUDA})

find_package(Python3 COMPONENTS Interpreter Development)
message("Found python: ${Python3_FOUND}, at ${Python3_LIBRARIES}")

find_package(PkgConfig)
pkg_search_module(LAPACKE REQUIRED lapacke)

if(NOT DEFINED MZ_PYTHONPATH)
  set(MZ_PYTHONPATH "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(MZ_PYTHONPATH "${MZ_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

set(ENV{PYTHONPATH} "${CMAKE_CURRENT_SOURCE_DIR}")

# Set the software version from version.txt file (if not already set)
if(NOT DEFINED SW_VERSION)
  file(STRINGS "packaging/version.txt" SW_VERSION)
  message(STATUS "Set SW_VERSION = ${SW_VERSION} from ${CMAKE_CURRENT_SOURCE_DIR}/packaging/version.txt")
else()
  message(STATUS "SW_VERSION already set to ${SW_VERSION}.")
endif()

# Set the deployment paths
set(ZOO_INSTALL_DIR "${CMAKE_BINARY_DIR}/staging")

# ----------------------------------
# Conditional build for TensorFlow
# ----------------------------------
if (NOT (DEFINED ENABLE_TENSORFLOW))
  message("AIMET Model Zoo TensorFlow build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TENSORFLOW ON CACHE BOOL "")
endif(NOT (DEFINED ENABLE_TENSORFLOW))

if (ENABLE_TENSORFLOW)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aimet_zoo_tensorflow
      DESTINATION ${ZOO_INSTALL_DIR}
      PATTERN "__pycache*" EXCLUDE
      PATTERN "CMakeLists.txt" EXCLUDE
      PATTERN "*.egg-info" EXCLUDE
      )
else (ENABLE_TENSORFLOW)
    message("AIMET TensorFlow build disabled")    
endif (ENABLE_TENSORFLOW)


# ----------------------------------
# Conditional build for PyTorch
# ----------------------------------
if (NOT (DEFINED ENABLE_TORCH))
  message("AIMET Model Zoo TensorFlow build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TORCH ON CACHE BOOL "")
endif(NOT (DEFINED ENABLE_TORCH))

if (ENABLE_TORCH)
    # Install the source directories
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aimet_zoo_torch
        DESTINATION ${ZOO_INSTALL_DIR}
        PATTERN "__pycache*" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN "*.egg-info" EXCLUDE
        )

else (ENABLE_TORCH)
    message("AIMET PyTorch build disabled")    
endif (ENABLE_TORCH)


# -------------------------------
# Generate pip packages
# -------------------------------

# Set the packaging path (if not already set)
if(NOT DEFINED AIMET_PACKAGE_PATH)
    set(AIMET_PACKAGE_PATH "\"${ZOO_INSTALL_DIR}\"")
    message(STATUS "Set AIMET_PACKAGE_PATH = ${AIMET_PACKAGE_PATH}")
endif(NOT DEFINED AIMET_PACKAGE_PATH)

add_custom_target(packagemodelzoo
    # Run the install target first
    COMMAND "${CMAKE_COMMAND}" --build . --target install

    # Now run the packaging target to generate wheel files
    COMMAND ${CMAKE_COMMAND} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -DSW_VERSION=${SW_VERSION} -DPROJECT_NAME=${CMAKE_PROJECT_NAME} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_model_zoo.cmake
)

# -------------------------------
# Code Violations
# -------------------------------

add_custom_target(pylintmodelzoo
    # Run the Pylint
    COMMAND ${CMAKE_COMMAND} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -DSW_VERSION=${SW_VERSION} -DPROJECT_NAME=${CMAKE_PROJECT_NAME} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/pylint_model_zoo.cmake
)

# -------------------------------
# Acceptance tests 
# -------------------------------

add_subdirectory(AcceptanceTests)


# -------------------------------
# Deployment
# -------------------------------

# Check the deployment path and set it to a default value (if not set)
if(NOT DEFINED DEPLOYMENT_PATH)
    set(DEPLOYMENT_PATH "./deploy")
    message(WARNING "DEPLOYMENT_PATH was not specified. Setting it to ${DEPLOYMENT_PATH}.")
else()
    message(STATUS "DEPLOYMENT_PATH already set to ${DEPLOYMENT_PATH}.")
endif()

add_custom_target(deploy)

set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")
file(GLOB package_tar_file "${build_packaging_dir}/*.tar.gz")
add_custom_command(TARGET deploy
    COMMAND test -d ${DEPLOYMENT_PATH} || echo "Deployment directory ${DEPLOYMENT_PATH} does NOT exist. Creating..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOYMENT_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy ${package_tar_file} ${DEPLOYMENT_PATH}
    )

# -------------------------------
# Upload pip packages
# -------------------------------

# Check the pip config file path and set it to a default value (if not set)
if(NOT DEFINED PIP_CONFIG_FILE)
    set(PIP_CONFIG_FILE "None")
endif()

# Check the pip index name and set it to a default value (if not set)
if(NOT DEFINED PIP_INDEX)
    set(PIP_INDEX "None")
endif()

# Check the pip certificate path and set it to a default value (if not set)
if(NOT DEFINED PIP_CERT_FILE)
    set(PIP_CERT_FILE "None")
endif()

#TODO For some reason, this package upload target does NOT work as expected and needs to be debugged
add_custom_target(upload
    # Now run the packaging target to upload the pip package
    COMMAND ${CMAKE_COMMAND} -DPIP_CONFIG_FILE=${PIP_CONFIG_FILE} -DPIP_INDEX=${PIP_INDEX} -DPIP_CERT_FILE=${PIP_CERT_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/upload_model_zoo.cmake
)

add_dependencies(upload packagemodelzoo)