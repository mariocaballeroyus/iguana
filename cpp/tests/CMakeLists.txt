add_executable(iguana_tests
    assembly/test_assembler.cpp
    basis/test_bspline.cpp
    basis/test_tensor_product.cpp
    domain/test_tensor_domain.cpp
    geometry/test_differential.cpp
    geometry/test_patch.cpp
    quadrature/test_gauss_legendre.cpp)

target_link_libraries(iguana_tests PRIVATE iguana Catch2::Catch2WithMain)

include(Catch)

catch_discover_tests(iguana_tests)
