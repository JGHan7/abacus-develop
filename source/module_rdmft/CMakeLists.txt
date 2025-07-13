# if(ENABLE_LCAO)
#     add_library(
#         rdmft
#         OBJECT
#         rdmft.cpp
#         rdmft_tools.cpp
#         rdmft_cal_V.cpp
#         rdmft_pot.cpp
#         rdmft_update_state.cpp
#         esolver_rdmft.cpp
#         optimizer/iter_diag_NOs.cpp
#         optimizer/optimizer_tools.cpp
#         optimizer/line_search_rdmft.cpp
#         optimizer/bfgs_opti_ONs.cpp
#         optimizer/parameterize_ONs/param_ONs.cpp
#         optimizer/parameterize_ONs/ebi_constraint.cpp
#         optimizer/idmft.cpp
#         optimizer/mixing_dmk.cpp
#         optimizer/ft_rdmft.cpp
#         optimizer/parameterize_ONs/softmax.cpp
#     )

#     # target_compile_definitions(rdmft PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
#     # target_link_libraries(rdmft ${math_libs})

# endif()


if(ENABLE_LCAO)
    list(APPEND rdmft_objects
        rdmft.cpp
        rdmft_tools.cpp
        rdmft_cal_V.cpp
        rdmft_pot.cpp
        rdmft_update_state.cpp
        esolver_rdmft.cpp
        optimizer/iter_diag_NOs.cpp
        optimizer/optimizer_tools.cpp
        optimizer/line_search_rdmft.cpp
        optimizer/bfgs_opti_ONs.cpp
        optimizer/parameterize_ONs/param_ONs.cpp
        optimizer/parameterize_ONs/ebi_constraint.cpp
        optimizer/idmft.cpp
        optimizer/mixing_dmk.cpp
        optimizer/ft_rdmft.cpp
        optimizer/parameterize_ONs/softmax.cpp
    )

    add_library(
        rdmft
        OBJECT
        ${rdmft_objects}
    )

endif()

