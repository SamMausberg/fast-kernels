#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native build metadata for fast-kernels.";
  m.def("build_info", []() {
    py::dict info;
    info["project_version"] = std::string(FK_PROJECT_VERSION);
    info["compiled_with_cuda"] = static_cast<bool>(FK_COMPILED_WITH_CUDA);
    info["cxx_compiler_id"] = std::string(FK_CXX_COMPILER_ID);
    info["cxx_compiler_version"] = std::string(FK_CXX_COMPILER_VERSION);
    info["cuda_compiler_version"] = std::string(FK_CUDA_COMPILER_VERSION);
    return info;
  });
}

