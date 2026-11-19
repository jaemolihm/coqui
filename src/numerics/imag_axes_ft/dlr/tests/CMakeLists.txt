
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID iaft_dlr)

set( TEST_SRCS
     test_dlr.cpp
   )

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

add_executable(test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST test_${TEST_ID} APPEND PROPERTY LABELS "iaft")

# The same suite on MKL's conditional-numerical-reproducibility path, where the cf2it solve rounds the Twt rows wn and
# -wn differently (issue #129). With another BLAS the variable is ignored and this repeats the suite.
add_mpi_unit_test(test_${TEST_ID}_mkl_cbwr_compatible "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" 1)
set_tests_properties(test_${TEST_ID}_mkl_cbwr_compatible PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                     ENVIRONMENT "OMP_NUM_THREADS=1;MKL_CBWR=COMPATIBLE")
set_property(TEST test_${TEST_ID}_mkl_cbwr_compatible APPEND PROPERTY LABELS "iaft")
