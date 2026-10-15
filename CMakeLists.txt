cmake_minimum_required(VERSION 3.21)

project(IGUANA VERSION 0.1.0 LANGUAGES CXX)

option(IGUANA_BUILD_PYTHON "Build the Python bindings" OFF)
option(IGUANA_WITH_PETSC "Assemble and solve with PETSc" ON)

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)

if(IGUANA_WITH_PETSC)
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_package(PkgConfig REQUIRED)

    if(DEFINED ENV{PETSC_DIR})
        string(CONCAT iguana_petsc_pkgconfig
               "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:"
               "$ENV{PKG_CONFIG_PATH}")
        set(ENV{PKG_CONFIG_PATH} "${iguana_petsc_pkgconfig}")
    endif()

    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
endif()

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
