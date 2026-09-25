add_library(iguana
    basis/bspline.cpp
    basis/tensor_bspline.cpp
    domain/tensor_domain.cpp
    domain/tensor_domain_iterator.cpp
    patch/patch.cpp
    quadrature/domain_quadrature.cpp
    quadrature/gauss_legendre.cpp)

add_library(IGUANA::iguana ALIAS iguana)

target_compile_features(iguana PUBLIC cxx_std_20)

set_target_properties(iguana PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(iguana PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

target_include_directories(iguana PUBLIC ${PROJECT_SOURCE_DIR}/cpp)

target_link_libraries(iguana PUBLIC Eigen3::Eigen)
