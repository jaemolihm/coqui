#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRC
  pgto.cpp
  rotate.cpp
  orbital_generator.cpp
  eph_vertex.cpp
)
ADD_LIBRARY(orbit ${SRC})
TARGET_LINK_LIBRARIES(orbit PUBLIC meanfield hamilt numerics)
install(TARGETS orbit
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

