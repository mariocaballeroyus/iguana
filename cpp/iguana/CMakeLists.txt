add_library(iguana basis/bspline.cpp)

target_compile_features(iguana PUBLIC cxx_std_20)

set_target_properties(iguana PROPERTIES CXX_EXTENSIONS OFF)

target_compile_options(iguana PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

target_include_directories(iguana PUBLIC ${PROJECT_SOURCE_DIR}/cpp)

target_link_libraries(iguana PUBLIC Eigen3::Eigen)
