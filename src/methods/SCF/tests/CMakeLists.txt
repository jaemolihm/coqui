
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_scf)

add_executable(test_${TEST_ID} test_simple_dyson.cpp test_scf_common.cpp test_thermodynamics.cpp test_lr_dyson.cpp test_mixing.cpp test_lr_gcr.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib scf_lib)


add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Column-block split of the Picard sweep (dyson_no_inversion): the fixture has 8 (s,k) pairs and the sweep uses
# floor(ranks / 8) column blocks per (s,k), so 16 ranks is the minimum that splits columns (2 blocks per (s,k))
if(CTEST_NPROC GREATER_EQUAL 16)
  add_mpi_unit_test(test_${TEST_ID}_dyson_picard_np16 "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" 16 "dyson_no_inversion_picard")
  set_tests_properties(test_${TEST_ID}_dyson_picard_np16 PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
endif()
