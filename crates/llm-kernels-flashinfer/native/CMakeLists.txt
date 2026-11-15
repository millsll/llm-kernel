cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(flashinfer_adapter LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)


find_package(CUDAToolkit REQUIRED)

# FlashInfer source dir
set(FLASHINFER_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/flashinfer"
)
# Check source dir
if(NOT EXISTS "${FLASHINFER_ROOT}/include/flashinfer")
    message(FATAL_ERROR
      "FlashInfer source not found: ${FLASHINFER_ROOT}"
    )
endif()


add_library(flashinfer_adapter SHARED
  src/normalization/rms_norm.cu
  src/attention/single_prefill.cu
)

target_include_directories(flashinfer_adapter BEFORE PRIVATE
    "${FLASHINFER_ROOT}/3rdparty/cccl/libcudacxx/include"
)

target_include_directories(flashinfer_adapter
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${FLASHINFER_ROOT}/include"
    "${FLASHINFER_ROOT}/3rdparty/cutlass/include"
)

target_link_libraries(flashinfer_adapter
  PRIVATE
    CUDA::cudart
)

set_target_properties(flashinfer_adapter PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

install(
  TARGETS flashinfer_adapter
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
