find_path(ProteoLizardAlgorithm_INCLUDE_DIR NAMES ProteoLizardAlgorithm/Hashing.h)
mark_as_advanced(ProteoLizardAlgorithm_INCLUDE_DIR)

find_library(ProteoLizardAlgorithm_LIBRARY NAMES proteolizardalgorithm)
mark_as_advanced(ProteoLizardAlgorithm_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ProteoLizardAlgorithm DEFAULT_MSG ProteoLizardAlgorithm_INCLUDE_DIR ProteoLizardAlgorithm_LIBRARY)

include(CMakeFindDependencyMacro)
find_dependency(Eigen3 REQUIRED)
find_dependency(ProteoLizardData REQUIRED)

if(ProteoLizardAlgorithm_FOUND)
    set(ProteoLizardAlgorithm_INCLUDE_DIRS ${ProteoLizardAlgorithm_INCLUDE_DIR})
    set(ProteoLizardAlgorithm_LIBRARIES    ${ProteoLizardAlgorithm_LIBRARY})
    if(NOT TARGET ProteoLizard::proteolizardalgorithm)
        include("${CMAKE_CURRENT_LIST_DIR}/ProteoLizardAlgorithmTargets.cmake")
    endif()
endif()
