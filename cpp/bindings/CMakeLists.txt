set(IGUANA_PYTHON_DIR ${PROJECT_BINARY_DIR}/python/iguana)

pybind11_add_module(iguana_python module.cpp basis.cpp patch.cpp
                    boundary.cpp grid.cpp embedding.cpp fspace.cpp
                    quadrature.cpp)

set_target_properties(iguana_python PROPERTIES
    OUTPUT_NAME cpp
    LIBRARY_OUTPUT_DIRECTORY ${IGUANA_PYTHON_DIR})

target_compile_features(iguana_python PRIVATE cxx_std_20)

target_compile_definitions(iguana_python PRIVATE
    IGUANA_VERSION="${PROJECT_VERSION}")

target_compile_options(iguana_python PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/MP>)

target_link_libraries(iguana_python PRIVATE IGUANA::iguana)

# The assembly hands its PETSc matrices to Python through petsc4py, which
# must wrap the PETSc that IGUANA links, as two of them in one process crash
if(IGUANA_WITH_PETSC)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import petsc4py; \
c = petsc4py.get_config(); \
print(petsc4py.get_include(), c['PETSC_DIR'] + '/' + c['PETSC_ARCH'], \
      sep=';', end='')"
        OUTPUT_VARIABLE iguana_petsc4py
        RESULT_VARIABLE iguana_petsc4py_status)

    if(NOT iguana_petsc4py_status EQUAL 0)
        message(FATAL_ERROR "IGUANA_WITH_PETSC needs petsc4py")
    endif()

    list(GET iguana_petsc4py 0 iguana_petsc4py_include)
    list(GET iguana_petsc4py 1 iguana_petsc4py_petsc)
    file(REAL_PATH ${iguana_petsc4py_petsc} iguana_petsc4py_petsc)
    file(REAL_PATH ${PETSc_PREFIX} iguana_petsc)

    if(NOT iguana_petsc4py_petsc STREQUAL iguana_petsc)
        message(FATAL_ERROR "petsc4py wraps the PETSc in "
                            "${iguana_petsc4py_petsc}, not the one in "
                            "${iguana_petsc} that IGUANA links")
    endif()

    target_sources(iguana_python PRIVATE assembly.cpp)

    target_include_directories(iguana_python PRIVATE
        ${iguana_petsc4py_include})
endif()

if(SKBUILD)
    install(TARGETS iguana_python DESTINATION ${SKBUILD_PROJECT_NAME})
else()
    add_custom_target(iguana_package ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${PROJECT_SOURCE_DIR}/python/iguana ${IGUANA_PYTHON_DIR}
        COMMENT "Staging the Python package")

    add_dependencies(iguana_package iguana_python)
endif()

if(BUILD_TESTING)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE iguana_pytest_status
        OUTPUT_QUIET ERROR_QUIET)

    if(iguana_pytest_status EQUAL 0)
        add_test(NAME iguana_python
                 COMMAND ${Python_EXECUTABLE} -m pytest -q
                         ${PROJECT_SOURCE_DIR}/python/tests)

        set_tests_properties(iguana_python PROPERTIES
            ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")
    endif()
endif()
