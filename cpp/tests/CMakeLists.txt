add_executable(iguana_tests basis/test_bspline.cpp)

target_compile_features(iguana_tests PRIVATE cxx_std_20)

target_link_libraries(iguana_tests PRIVATE
    IGUANA::iguana
    Catch2::Catch2WithMain)

add_test(NAME iguana_tests COMMAND iguana_tests)
