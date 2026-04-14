cmake_minimum_required(VERSION 4.2)

project(
  fast_kernels
  VERSION 0.1.0
  DESCRIPTION "Benchmark-first CUDA kernel experiments"
  LANGUAGES CXX
)

option(FK_ENABLE_CUDA "Build CUDA sources and link against the CUDA runtime." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(FK_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

pybind11_add_module(
  _native
  MODULE
  csrc/bindings/python_module.cpp
)

target_include_directories(_native PRIVATE csrc/include)
target_include_directories(_native PRIVATE csrc)
target_compile_features(_native PRIVATE cxx_std_20)
target_compile_definitions(
  _native
  PRIVATE
    FK_PROJECT_VERSION="${PROJECT_VERSION}"
    FK_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    FK_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

if(FK_ENABLE_CUDA)
  target_sources(
    _native
    PRIVATE
      csrc/kernels/template/noop.cu
      csrc/kernels/decode_quant_linear/arc_w4a16.cu
  )
  target_link_libraries(_native PRIVATE CUDA::cudart CUDA::cublasLt CUDA::cublas)
  target_compile_definitions(
    _native
    PRIVATE
      FK_COMPILED_WITH_CUDA=1
      FK_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
  )
else()
  target_compile_definitions(
    _native
    PRIVATE
      FK_COMPILED_WITH_CUDA=0
      FK_CUDA_COMPILER_VERSION=""
  )
endif()

install(TARGETS _native LIBRARY DESTINATION fast_kernels)
