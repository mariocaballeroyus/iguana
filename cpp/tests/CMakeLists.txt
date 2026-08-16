add_executable(iguana_tests
    basis/test_bspline.cpp
    basis/test_tensor_product.cpp)

target_link_libraries(iguana_tests PRIVATE iguana Catch2::Catch2WithMain)

include(Catch)

catch_discover_tests(iguana_tests)
