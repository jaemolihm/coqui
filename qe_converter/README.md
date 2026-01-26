Quantum ESPRESSO Converter for CoQuí
------------------------------------
**Last Updated:** Jan. 25, 2026

This directory provides the Fortran source code `pw2coqui.f90` required to 
interface **Quantum ESPRESSO (QE)** with **CoQuí**. The converter extracts 
the necessary data from QE calculations and makes it available for use in 
CoQuí workflows.

## 📦 Installation

The converter is distributed with the CoQuí source code, but it is **not 
compiled as part of CoQuí itself**. Instead, you need to integrate it into 
your local QE source tree and compile it together with QE.

### Step 1: Locate the converter source files
Copy `pw2coqui.f90` into the **`PP/src/` folder** (the *PostProc* package in the
QE suite) of your QE source tree:  
```bash
cp -r pw2coqui.f90 /path/to/qe-source/PP/src/
```

### Step 2: Modify QE's build system

Choose the appropriate option based on your QE build system:

#### Option A: CMake

Edit `PP/CMakeLists.txt` in your QE source tree:

1. Insert the following block above the `PP_EXE_TARGETS` definition:
   ```cmake
   ###########################################################
   # pw2coqui.x
   ###########################################################
   set(src_pw2coqui_x src/pw2coqui.f90)
   qe_add_executable(qe_pp_pw2coqui_exe ${src_pw2coqui_x})
   set_target_properties(qe_pp_pw2coqui_exe PROPERTIES OUTPUT_NAME pw2coqui.x)
   target_link_libraries(qe_pp_pw2coqui_exe
       PRIVATE
           qe_pw
           qe_modules
           qe_pp
           qe_upflib
           qe_fftx
           qe_mpi_fortran
           qe_xclib)
   ```

2. Add the new executable target to `PP_EXE_TARGETS`:
   ```cmake
   set(PP_EXE_TARGETS
       ...
       qe_pp_pw2coqui_exe
       ...
   )
   ```

#### Option B: Autotools (Makefile)

Edit `PP/src/Makefile` in your QE source tree:

1. Add `pw2coqui.x` to the `all` target list:
   ```makefile
   all : tldeps open_grid.x average.x ... pw2coqui.x
   ```

2. Add the build rule (e.g., before the `clean` target):
   ```makefile
   pw2coqui.x : pw2coqui.o libpp.a $(MODULES)
      $(LD) $(LDFLAGS) -o $@ \
         pw2coqui.o libpp.a $(MODULES) $(QELIBS)
      - ( cd ../../bin ; ln -fs ../PP/src/$@ . )
   ```

### Step 3: Recompile QE

Rebuild QE with the modified configuration:
- **CMake**: Run `make` from your CMake build directory
- **Autotools**: Run `make pp` from the QE root directory

After successful compilation, the executable `pw2coqui.x` will be available
in your QE `bin/` directory.


## Usage
Once QE has been recompiled with the converter, you can proceed with the 
CoQuí [Quickstart tutorials](https://github.com/AbInitioQHub/coqui-tutorial/blob/main/quickstart/01s_dft_to_coqui_converter.ipynb) 
to generate CoQuí inputs from QE.