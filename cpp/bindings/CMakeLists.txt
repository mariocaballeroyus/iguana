set(IGUANA_PYTHON_DIR ${PROJECT_BINARY_DIR}/python/iguana)

pybind11_add_module(iguana_python module.cpp basis.cpp patch.cpp domain.cpp
                    quadrature.cpp)

set_target_properties(iguana_python PROPERTIES
    OUTPUT_NAME cpp
    LIBRARY_OUTPUT_DIRECTORY ${IGUANA_PYTHON_DIR})

target_compile_features(iguana_python PRIVATE cxx_std_20)

target_compile_definitions(iguana_python PRIVATE
    IGUANA_VERSION="${PROJECT_VERSION}")

target_compile_options(iguana_python PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

target_link_libraries(iguana_python PRIVATE IGUANA::iguana)

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
