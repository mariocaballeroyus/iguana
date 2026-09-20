cmake_minimum_required(VERSION 3.21)

project(IGUANA VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(external/catch2)
    add_subdirectory(cpp/tests)
endif()
