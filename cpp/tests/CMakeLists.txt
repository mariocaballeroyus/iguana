add_executable(iguana_tests
    basis/test_bspline.cpp
    basis/test_tensor_bspline.cpp
    domain/test_hierarchical_domain.cpp
    domain/test_knot_vector.cpp
    domain/test_tensor_domain_iterator.cpp
    embedding/test_clipper.cpp
    embedding/test_inside.cpp
    embedding/test_slicer.cpp
    geometry/test_patch.cpp
    quadrature/test_domain_quadrature.cpp
    quadrature/gauss_legendre/test_gauss_legendre.cpp
    quadrature/moment_fitting/test_moment_fitting.cpp
    quadrature/moment_fitting/test_moments.cpp
    quadrature/xiao_gimbutas/test_xiao_gimbutas.cpp
    utils/test_legendre.cpp)

target_compile_features(iguana_tests PRIVATE cxx_std_20)

target_link_libraries(iguana_tests PRIVATE
    IGUANA::iguana
    Catch2::Catch2WithMain)

add_test(NAME iguana_tests COMMAND iguana_tests)
