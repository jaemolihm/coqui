
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID math_distributed_nda)

if(ENABLE_SLATE)
set( TEST_SRCS 
     test_distributed.cpp
     test_slate.cpp
   )
else(ENABLE_SLATE)
set( TEST_SRCS 
     test_distributed.cpp
   )
endif(ENABLE_SLATE)

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
if(ENABLE_SLATE)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils nda_c h5_c iaft_utils_lib slate_lapack_api slate)
else(ENABLE_SLATE)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils nda_c h5_c)
endif(ENABLE_SLATE)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# The tiling arms of this binary are grid-dependent, and a partition bug can be
# invisible at one rank count and fatal at another: the ragged-partition failures were
# found at 7, 11, 13 and 14 while CTEST_NPROC = 8 stayed green throughout. Register the
# same binary at three prime counts so the property is actually covered.
#
# Their label is "unit-sweep", NOT "unit", so `ctest -L unit` keeps its current cost;
# run the sweep explicitly with `ctest -L unit-sweep`. A count above the slot count
# CMake detected is not registered at all: the mpi dependency forces --oversubscribe
# into MPIEXEC_PREFLAGS for Open MPI, so such a run would time-slice the host rather
# than fail, and other MPIs fail the launch outright. Neither is a test result.
foreach(SWEEP_NPROC 7 11 13)
  if(SWEEP_NPROC GREATER MPIEXEC_MAX_NUMPROCS)
    message(STATUS "Skipping test_${TEST_ID}_np${SWEEP_NPROC}: only ${MPIEXEC_MAX_NUMPROCS} MPI slots detected")
    continue()
  endif()
  add_mpi_unit_test(test_${TEST_ID}_np${SWEEP_NPROC}
                    "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${SWEEP_NPROC})
  set_tests_properties(test_${TEST_ID}_np${SWEEP_NPROC} PROPERTIES
                       WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
  set_property(TEST test_${TEST_ID}_np${SWEEP_NPROC} PROPERTY LABELS "unit-sweep")
endforeach()
