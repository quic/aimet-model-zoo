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

add_custom_target( AcceptanceTests.Torch )

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/test_resnet_quanteval.py")
    foreach(filename ${files})

        get_filename_component( testname "${filename}" NAME_WE )
        message(STATUS "Testname: " testname)

        add_custom_target(AcceptanceTests.Torch.${testname}
            VERBATIM COMMAND ${CMAKE_COMMAND} -E env
                             ${MZ_PYTHONPATH} 
                             ${Python3_EXECUTABLE} -m pytest -s ${filename} 
                             ${CUDA_FLAG} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output_${testname}.xml) 

        add_dependencies( AcceptanceTests.Torch AcceptanceTests.Torch.${testname} )
    endforeach( filename )