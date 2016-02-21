cmake_minimum_required(VERSION 3.3)
project(diploma)

set(CMAKE_VERBOSE_MAKEFILE 1)
set(CMAKE_CXX_COMPILER "/usr/local/bin/clang-omp++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(diploma ${SOURCE_FILES})