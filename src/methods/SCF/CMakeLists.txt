#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(scf_lib simple_dyson.cpp dca_dyson.cpp dca_dyson.h scf_driver.hpp scf_driver.cpp rpa.cpp scf_common.hpp scf_common.cpp qp_scf_common.cpp mb_solver_t.h qp_params_t.h lr_state.hpp lr_dyson.hpp lr_dyson.cpp lr_diis.hpp lr_driver.hpp lr_driver.cpp lr_energy_curvature.hpp lr_energy_curvature.cpp lr_precompute.hpp lr_ibc.hpp scf_mem_report.hpp scf_mem_report.cpp)
target_link_libraries(scf_lib PUBLIC utils numerics meanfield hamilt ac_lib hf_lib gw_lib gf2_lib methods_tools_lib)
install(TARGETS scf_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
