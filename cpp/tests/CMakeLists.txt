add_executable(iguana_tests smoke.cpp)

target_compile_features(iguana_tests PRIVATE cxx_std_20)

target_include_directories(iguana_tests PRIVATE ${PROJECT_SOURCE_DIR}/cpp)

target_link_libraries(iguana_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME iguana_tests COMMAND iguana_tests)
