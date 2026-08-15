cmake_minimum_required(VERSION 3.21)

project(IGUANA VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)

add_subdirectory(cpp/iguana)

enable_testing()

add_subdirectory(external/catch2 EXCLUDE_FROM_ALL)

add_subdirectory(cpp/tests)
