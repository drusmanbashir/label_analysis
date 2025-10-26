cmake_minimum_required (VERSION 3.16)

project(printcpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Prefer system pybind11 if you have it; otherwise add_subdirectory(pybind11) here.
find_package(pybind11 REQUIRED)

pybind11_add_module(printcpp MODULE printmod.cpp)

target_link_libraries(printcpp PRIVATE ${ITK_LIBRARIES})

# Make import name clean (`import labelgeometry_cpp`)
set_target_properties(printcpp PROPERTIES
  OUTPUT_NAME "printcpp"
  PREFIX ""  # no lib prefix on Linux
)

# Helpful on Linux to reduce symbol visibility
