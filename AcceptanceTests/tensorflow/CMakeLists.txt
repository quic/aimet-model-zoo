# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
if (ENABLE_CUDA)
    set(CUDA_FLAG -m "not blah")
    set(USE_CUDA True)
else (ENABLE_CUDA)
    set(CUDA_FLAG -m "not cuda")
    set(USE_CUDA False)
endif (ENABLE_CUDA)

add_custom_target( AcceptanceTests.Tensorflow )

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*quanteval.py")
    foreach(filename ${files})
        get_filename_component( testname "${filename}" NAME_WE )
        string(SUBSTRING ${testname} 5 -1 formatted_name)
        message(STATUS  "Now running Tensorflow Test: ${formatted_name}")
        
        add_custom_target(AcceptanceTests.Tensorflow.${testname}
            VERBATIM COMMAND . /opt/${formatted_name}-venv/bin/activate && export HF_HOME=/opt/cache &&
                             ${CMAKE_COMMAND} -E env
                             ${MZ_PYTHONPATH}:$$PYTHONPATH 
                             python -m pytest -s ${filename} 
                             ${CUDA_FLAG} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output_${testname}.xml) 

        add_dependencies( AcceptanceTests.Tensorflow AcceptanceTests.Tensorflow.${testname} )
    endforeach( filename )