cmake_minimum_required(VERSION 3.16)
project(neuron-shim VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------ #
# Options                                                              #
# ------------------------------------------------------------------ #
option(SHIM_ENABLE_TFLITE  "Build TFLite backend"       ON)
option(SHIM_ENABLE_ONNX    "Build ONNX Runtime backend" ON)
option(SHIM_ENABLE_GPU     "Enable TFLite GPU delegate" OFF)
option(SHIM_BUILD_TESTS    "Build test programs"         ON)

# ------------------------------------------------------------------ #
# Core shim library (always built)                                     #
# ------------------------------------------------------------------ #
add_library(neuron_shim SHARED
    src/shim_runtime.c
    src/config.c
    src/model_resolver.c
    src/backend_stub.c
    src/backend_selector.c
)

target_include_directories(neuron_shim PUBLIC include)
target_link_libraries(neuron_shim PRIVATE dl pthread)

# Alias as libneuronrt.so for direct replacement
set_target_properties(neuron_shim PROPERTIES
    OUTPUT_NAME "neuronrt"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ------------------------------------------------------------------ #
# libapusys stub                                                       #
# ------------------------------------------------------------------ #
add_library(apusys_shim SHARED
    src/shim_apusys.c
)
set_target_properties(apusys_shim PROPERTIES
    OUTPUT_NAME "apusys"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ------------------------------------------------------------------ #
# TFLite backend (optional)                                            #
# ------------------------------------------------------------------ #
if(SHIM_ENABLE_TFLITE)
    # Look for TFLite C API
    find_library(TFLITE_LIB tensorflowlite_c
        HINTS
            /usr/lib
            /usr/local/lib
            ${CMAKE_SYSROOT}/usr/lib
            $ENV{TFLITE_DIR}/lib
    )

    find_path(TFLITE_INCLUDE tensorflow/lite/c/c_api.h
        HINTS
            /usr/include
            /usr/local/include
            $ENV{TFLITE_DIR}/include
    )

    if(TFLITE_LIB AND TFLITE_INCLUDE)
        target_sources(neuron_shim PRIVATE src/backend_tflite.c)
        target_include_directories(neuron_shim PRIVATE ${TFLITE_INCLUDE})
        target_link_libraries(neuron_shim PRIVATE ${TFLITE_LIB})
        target_compile_definitions(neuron_shim PRIVATE SHIM_HAS_TFLITE=1)

        if(SHIM_ENABLE_GPU)
            target_compile_definitions(neuron_shim PRIVATE NEURON_SHIM_ENABLE_GPU=1)
        endif()

        message(STATUS "TFLite backend: ENABLED (${TFLITE_LIB})")
    else()
        message(STATUS "TFLite backend: DISABLED (library not found)")
        message(STATUS "  Set TFLITE_DIR or install libtensorflowlite_c")
    endif()
else()
    message(STATUS "TFLite backend: DISABLED (by option)")
endif()

# ------------------------------------------------------------------ #
# ONNX Runtime backend (optional — preferred for GPU inference)        #
# ------------------------------------------------------------------ #
if(SHIM_ENABLE_ONNX)
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS
            /usr/lib
            /usr/local/lib
            ${CMAKE_SYSROOT}/usr/lib
            $ENV{ONNXRUNTIME_DIR}/lib
    )

    find_path(ONNXRUNTIME_INCLUDE onnxruntime/core/session/onnxruntime_c_api.h
        HINTS
            /usr/include
            /usr/local/include
            $ENV{ONNXRUNTIME_DIR}/include
    )

    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        target_sources(neuron_shim PRIVATE src/backend_onnx.c)
        target_include_directories(neuron_shim PRIVATE ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(neuron_shim PRIVATE ${ONNXRUNTIME_LIB})
        target_compile_definitions(neuron_shim PRIVATE SHIM_HAS_ONNX=1)
        message(STATUS "ONNX Runtime backend: ENABLED (${ONNXRUNTIME_LIB})")
        message(STATUS "  GPU support depends on ORT build:")
        message(STATUS "    NVIDIA: install onnxruntime-gpu (CUDA/TensorRT EP)")
        message(STATUS "    AMD:    install onnxruntime-rocm (MIGraphX EP)")
    else()
        message(STATUS "ONNX Runtime backend: DISABLED (library not found)")
        message(STATUS "  Set ONNXRUNTIME_DIR or install libonnxruntime-dev")
    endif()
else()
    message(STATUS "ONNX Runtime backend: DISABLED (by option)")
endif()

# ------------------------------------------------------------------ #
# Test / trace program                                                 #
# ------------------------------------------------------------------ #
if(SHIM_BUILD_TESTS)
    add_executable(shim_test tests/test_basic.c)
    target_include_directories(shim_test PRIVATE include)
    target_link_libraries(shim_test PRIVATE neuron_shim)
endif()

# ------------------------------------------------------------------ #
# Install                                                              #
# ------------------------------------------------------------------ #
install(TARGETS neuron_shim apusys_shim
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/neuron-shim)

# ------------------------------------------------------------------ #
# Summary                                                              #
# ------------------------------------------------------------------ #
message(STATUS "")
message(STATUS "=== neuron-shim build configuration ===")
message(STATUS "  Install prefix : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Output names   : libneuronrt.so, libapusys.so")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  LD_PRELOAD=libneuronrt.so:libapusys.so ./target_binary")
message(STATUS "  NEURON_SHIM_BACKEND=stub  (for tracing)")
message(STATUS "  NEURON_SHIM_BACKEND=tflite (for real inference)")
message(STATUS "")
