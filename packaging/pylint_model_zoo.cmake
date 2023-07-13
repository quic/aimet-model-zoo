# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# CMake file to pylint AIMET model zoo
message("Preparing Directories for Pylint Results" ...)
set(src_results_dir "${SOURCE_DIR}/packaging/results")
file(MAKE_DIRECTORY "${SOURCE_DIR}/packaging/results")
message(STATUS "Source Directory: ${SOURCE_DIR}")
message(STATUS "Result Directory: ${src_results_dir}")

set(XDG_CACHE_HOME ${SOURCE_DIR})

if(ENABLE_TENSORFLOW)
    # Add AIMET Tensorflow package to package array list
    list(APPEND pylint_list "aimet_zoo_tensorflow")
endif()

if(ENABLE_TORCH)
    # Add AIMET Torch package to package array list
    list(APPEND pylint_list "aimet_zoo_torch")
endif()

message(STATUS "Linting to proceed for: ${pylint_list}")

set(MSG_FORMAT "--msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}'")

set(pylint_failed 0)
foreach(zoo_pylint_folder IN LISTS pylint_list)
    message(STATUS "Aimet Model Zoo Variant: ${zoo_pylint_folder}")
    message(STATUS "Aimet Model Zoo Path Verification: ${SOURCE_DIR}/${zoo_pylint_folder}")
    message(STATUS "Sending message format: ${MSG_FORMAT}")

    execute_process (
       COMMAND bash -c "pylint --rcfile=${SOURCE_DIR}/.pylintrc -r n ${MSG_FORMAT} ${SOURCE_DIR}/${zoo_pylint_folder}/ >> ${src_results_dir}/pylint_${zoo_pylint_folder} && \
       echo 'Pylinting...' && cat ${src_results_dir}/pylint_${zoo_pylint_folder}"
       RESULT_VARIABLE pylint_return
        )
    message(STATUS "Return Code is: ${pylint_return}")

    if(${pylint_return} AND NOT ${pylint_return} EQUAL "0")
       if(${pylint_return} EQUAL "1")
          message(STATUS "Return Code is: ${pylint_return} -- fatal message issued")
       elseif(${pylint_return} EQUAL "2")
          message(STATUS "Return Code is: ${pylint_return} -- error message issued")
       elseif(${pylint_return} EQUAL "4")
          message(STATUS "Return Code is: ${pylint_return} -- warning message issued")
       elseif(${pylint_return} EQUAL "8")
          message(STATUS "Return Code is: ${pylint_return} -- refactor message issued")
       elseif(${pylint_return} EQUAL "16")
          message(STATUS "Return Code is: ${pylint_return} -- convention message issued")
       elseif(${pylint_return} EQUAL "32")
          message(STATUS "Return Code is: ${pylint_return} -- usage error")
       else()
          message(STATUS "Return Code is: ${pylint_return} -- multiple messages issued")
       endif()

       set(pylint_failed 1)
    endif()

endforeach()
