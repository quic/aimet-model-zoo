
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

add_custom_target(AcceptanceTests)

if (ENABLE_TORCH)
    add_dependencies(AcceptanceTests
            AcceptanceTests.Torch)

endif (ENABLE_TORCH)

#if (ENABLE_TENSORFLOW)
#    add_dependencies(AcceptanceTests
#            AcceptanceTests.Tensorflow)
#
#endif (ENABLE_TENSORFLOW)

if (ENABLE_TORCH)
    add_subdirectory(torch)
endif (ENABLE_TORCH)

#if (ENABLE_TENSORFLOW)
#    add_subdirectory(tensorflow)
#endif (ENABLE_TENSORFLOW)
