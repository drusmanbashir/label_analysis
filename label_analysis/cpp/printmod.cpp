#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

static void say(const std::string& msg) {
    std::cout << msg << std::endl;  // endl flushes
}

PYBIND11_MODULE(printcpp, m) {
    m.doc() = "Minimal print-from-C++ module";
    m.def("say", &say, py::arg("msg"));
}
