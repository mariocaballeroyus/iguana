pybind11_add_module(iguana_python module.cpp)

set_target_properties(iguana_python PROPERTIES OUTPUT_NAME iguana)

target_compile_features(iguana_python PRIVATE cxx_std_20)

target_compile_definitions(iguana_python PRIVATE
    IGUANA_VERSION="${PROJECT_VERSION}")

target_compile_options(iguana_python PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

target_link_libraries(iguana_python PRIVATE IGUANA::iguana)
