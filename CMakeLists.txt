cmake_minimum_required(VERSION 4.2)

project(
  fast_kernels
  VERSION 0.1.0
  DESCRIPTION "Benchmark-first CUDA kernel experiments"
  LANGUAGES CXX
)

option(FK_ENABLE_CUDA "Build CUDA sources and link against the CUDA runtime." ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE FK_PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE FK_PYBIND11_STATUS
    ERROR_QUIET
  )
  if(FK_PYBIND11_STATUS EQUAL 0)
    set(pybind11_DIR "${FK_PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake package directory")
  endif()
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(FK_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  string(REPLACE ";" "," FK_CUDA_ARCHITECTURES_STR "${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(FK_CUDA_ARCHITECTURES_STR "")
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
    FK_CUDA_ARCHITECTURES="${FK_CUDA_ARCHITECTURES_STR}"
)

if(FK_ENABLE_CUDA)
  target_sources(
    _native
    PRIVATE
      csrc/kernels/template/noop.cu
      csrc/kernels/decode_quant_linear/arc_w4a16.cu
      csrc/kernels/clustered_page_decode/clustered_page_decode.cu
      csrc/kernels/prefix_union_decode/prefix_union_decode.cu
      csrc/kernels/rdkng/explicit_sketch.cu
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
