cmake_minimum_required(VERSION 3.21)

project(IGUANA VERSION 0.1.0 LANGUAGES CXX)

option(IGUANA_BUILD_PYTHON "Build the Python bindings" OFF)

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)
add_subdirectory(cpp/iguana)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(external/catch2)
    add_subdirectory(cpp/tests)
endif()

if(IGUANA_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON NEW)

    add_subdirectory(external/pybind11 EXCLUDE_FROM_ALL)
    add_subdirectory(cpp/bindings)
endif()
