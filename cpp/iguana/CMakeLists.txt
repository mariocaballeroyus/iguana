add_library(iguana
    assembly/assembler.cpp
    assembly/element_data.cpp
    basis/bspline.cpp
    basis/tensor_product.cpp
    domain/tensor_domain.cpp
    geometry/differential.cpp
    geometry/patch.cpp
    quadrature/gauss_legendre.cpp)

target_compile_features(iguana PUBLIC cxx_std_20)

set_target_properties(iguana PROPERTIES CXX_EXTENSIONS OFF)

target_compile_options(iguana PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

target_include_directories(iguana PUBLIC ${PROJECT_SOURCE_DIR}/cpp)

target_link_libraries(iguana PUBLIC Eigen3::Eigen)
