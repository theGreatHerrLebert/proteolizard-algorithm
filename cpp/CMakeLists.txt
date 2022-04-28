cmake_minimum_required(VERSION 3.16)

project(ProteoLizardAlgorithm
    VERSION      0.1
    DESCRIPTION  "Algorithms and tooling to process ion-mobility mass-spectrometry raw-data"
    HOMEPAGE_URL "https://github.com/theGreatHerrLebert/proteolizard-algorithm"
    LANGUAGES    CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(ProteoLizardData REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${ProteoLizardData_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_library(proteolizardalgorithm SHARED
    src/Hashing.cpp
    src/Proteolizardalgorithm.cpp
)
set(ProteoLizardAlgorithm_HEADER_FILES
    src/Hashing.h
)
set_target_properties(proteolizardalgorithm PROPERTIES PUBLIC_HEADER "${ProteoLizardAlgorithm_HEADER_FILES}")

target_link_libraries(proteolizardalgorithm
    PUBLIC
        ProteoLizard::proteolizarddata
        Eigen3::Eigen
)
