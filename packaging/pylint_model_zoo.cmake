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
file(MAKE_DIRECTORY results)

message(STATUS "Source Directory: ${SOURCE_DIR}")
message(STATUS "Result Directory: ${src_results_dir}")

set(zoo-pylint_folder_list "aimet_zoo_tensorflow;aimet_zoo_torch")
set(MSG_FORMAT "--msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}'")

# 
foreach(zoo_pylint_folder IN LISTS zoo-pylint_folder_list)
    message(STATUS "Aimet Model Zoo Variant: ${zoo_pylint_folder}")
    message(STATUS "Aimet Model Zoo Path Verification: ${SOURCE_DIR}/${zoo_pylint_folder}")
    execute_process (
       COMMAND pylint --rcfile=${SOURCE_DIR}/.pylintrc -r n ${MSG_FORMAT} ${SOURCE_DIR}/${zoo_pylint_folder}
       OUTPUT_VARIABLE pylint_complete
        )
    file(WRITE "${src_results_dir}/pylint_${zoo_pylint_folder}" ${pylint_complete})
    message(${pylint_complete})
endforeach()
