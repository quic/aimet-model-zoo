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

# TODO: temporarily set a list containing both tf and torch
# This should move to being selectable betrween tf and/or torch 
set(zoo-pylint_folder_list "aimet_zoo_tensorflow;aimet_zoo_torch")
set(MSG_FORMAT "--msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}'")

set(pylint_failed 0)
foreach(zoo_pylint_folder IN LISTS zoo-pylint_folder_list)
    message(STATUS "Aimet Model Zoo Variant: ${zoo_pylint_folder}")
    message(STATUS "Aimet Model Zoo Path Verification: ${SOURCE_DIR}/${zoo_pylint_folder}")
    execute_process (
       COMMAND pylint --rcfile=${SOURCE_DIR}/.pylintrc -r n ${MSG_FORMAT} ${SOURCE_DIR}/${zoo_pylint_folder}
       OUTPUT_VARIABLE pylint_complete
       RESULT_VARIABLE pylint_return
        )
    message(STATUS "Return Code is: ${pylint_return} -- Please correct the errors below")
    if(${pylint_return} AND NOT ${pylint_return} EQUAL "0")
       message( WARNING "Pylint failed for ${zoo_pylint_folder}")
       set(pylint_failed 1)
    endif()
    file(WRITE "${src_results_dir}/pylint_${zoo_pylint_folder}" ${pylint_complete})
    message(${pylint_complete})
endforeach()
