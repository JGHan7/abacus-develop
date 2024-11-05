if(ENABLE_LCAO)
    add_library(
        rdmft
        OBJECT
        rdmft.cpp
        rdmft_tools.cpp
        cal_V_rdmft.cpp
        update_state_rdmft.cpp
        esolver_rdmft.cpp
        optimizer/iter_diag_NOs.cpp
    )
endif()

# if(ENABLE_COVERAGE)
#   add_coverage(psi)
#   add_coverage(psi_initializer)
# endif()

# if (BUILD_TESTING)
#   add_subdirectory(kernels/test)
#   add_subdirectory(test)
# endif()
