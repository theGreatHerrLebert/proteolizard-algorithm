cmake_minimum_required(VERSION 3.16)
project(src)

include_directories("../../../proteolizard-data/include/eigen")
include_directories("../../../proteolizard-data/cpp/src")
include_directories("../../../opentims/opentims++")

find_library(LIB_TBB NAMES tbb REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../python/")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(proteolizardalgorithm SHARED Proteolizardalgorithm.cpp Hashing.h)

target_link_libraries(proteolizardalgorithm ${LIB_TBB})
target_link_libraries(proteolizardalgorithm ${PYTHON_LIBRARIES})
target_link_libraries(proteolizardalgorithm zstd)
target_link_libraries(proteolizardalgorithm sqlite3)
