add_library(iguana
    basis/bspline.cpp
    basis/tensor_bspline.cpp
    basis/tensor_nurbs.cpp
    domain/hierarchical_domain.cpp
    domain/hierarchical_domain_iterator.cpp
    domain/knot_vector.cpp
    domain/tensor_domain.cpp
    domain/tensor_domain_iterator.cpp
    embedding/embedding.cpp
    fspace/dof_map.cpp
    fspace/function_space.cpp
    geometry/patch.cpp
    quadrature/box_rule.cpp
    quadrature/domain_quadrature.cpp
    quadrature/gauss_legendre/gauss_legendre.cpp
    quadrature/gauss_legendre/gauss_legendre_tables.cpp
    quadrature/moment_fitting/moment_fitting.cpp
    quadrature/moment_fitting/moments.cpp
    quadrature/xiao_gimbutas/xiao_gimbutas.cpp
    quadrature/xiao_gimbutas/xiao_gimbutas_tables.cpp)

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
