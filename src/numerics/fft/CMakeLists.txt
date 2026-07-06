#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(fft_src
   )

if(ENABLE_FFTW)
  set(fft_src 
       ${fft_src}
       fftw.cpp 
     )
endif()

if(ENABLE_CUDA)
  set(fft_src 
       ${fft_src}
       cufft.cpp 
     )
endif()


# Build shared under Python so the FFT symbols live in a single libfft_lib.so
# shared by all c2py modules. Gated on COQUI_PYTHON_SUPPORT like io_lib (see 56645d9).
if(COQUI_PYTHON_SUPPORT)
  add_library(fft_lib SHARED ${fft_src})
else()
  add_library(fft_lib ${fft_src})
endif()
target_link_libraries(fft_lib PUBLIC utils nda_c)
if(ENABLE_FFTW)
  target_link_libraries(fft_lib PUBLIC FFTW::Double) 
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft) 
endif()
install(TARGETS fft_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
