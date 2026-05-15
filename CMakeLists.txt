cmake_minimum_required(VERSION 3.12)
project(Genesis_1_3_Version4 CXX C)

option(USE_DPI "Use diagnostic plugin interface" OFF)
option(USE_CUDA "Enable optional CUDA acceleration for field solvers" OFF)
option(GENESIS_ENABLE_NVTX "Enable NVTX range annotations for Nsight Systems profiling" ON)
set(GENESIS_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures to build for, e.g. 70;80;90. Empty lets CMake/NVCC choose the default.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENTIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
        find_package(CUDAToolkit REQUIRED)
        set(GENESIS_CUDA_RUNTIME_LIBRARY CUDA::cudart)
        set(GENESIS_CUDA_CUFFT_LIBRARY CUDA::cufft)
        set(GENESIS_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    else()
        find_package(CUDA REQUIRED)
        set(GENESIS_CUDA_RUNTIME_LIBRARY ${CUDA_CUDART_LIBRARY})
        set(GENESIS_CUDA_CUFFT_LIBRARY ${CUDA_cufft_LIBRARY})
        if(NOT GENESIS_CUDA_CUFFT_LIBRARY)
            set(GENESIS_CUDA_CUFFT_LIBRARY ${CUDA_CUFFT_LIBRARIES})
        endif()
        set(GENESIS_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    endif()
    message(STATUS "CUDA acceleration enabled")
endif()
# set(CMAKE_INSTALL_RPATH "/usr/local/lib")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# developer options
set(CMAKE_VERBOSE_MAKEFILE ON)
add_compile_options(-g)

# OpenMPI: preprocessor macro to disable C++ interface (then we don't have to link libmpi_cxx)
add_compile_definitions(OMPI_SKIP_MPICXX)

list(APPEND CMAKE_PREFIX_PATH $ENV{FFTW_ROOT})

# container for needed librarires
list(APPEND _libraries)
if(GENESIS_ENABLE_NVTX)
    list(APPEND _libraries ${CMAKE_DL_LIBS})
endif()

# Check for macOS with MacPorts
if(EXISTS "/opt/local/bin/h5pcc")
  set(CMAKE_CXX_COMPILER /opt/local/bin/h5pcc)
endif()

if (CMAKE_CXX_COMPILER MATCHES  "h5pcc")
   Message(STATUS "h5pcc compiler wrapper found - No additional libraries needed")
else()
   Message(STATUS "h5pcc not found - looking for basic compiler")
   Message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
   Message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")

   # check for MPI
   find_package(MPI REQUIRED)
   include_directories(${MPI_INCLUDE_PATH})
   Message(STATUS "MPI Found")

   # and check for HDF5
   SET(HDF5_PREFER_PARALLEL TRUE)
   find_package(HDF5 REQUIRED)
   Message(STATUS "HDF5 Found")
   Message(STATUS "Version: ${HDF5_VERSION}")
   if (TARGET HDF5::HDF5)
#       list(APPEND _libraries MPI::MPI_C hdf5_mpich)  # needed for my local computer
       list(APPEND _libraries MPI::MPI_C HDF5::HDF5)
   else()
       list(APPEND _libraries MPI::MPI_C)
   endif()
   include_directories(${HDF5_C_INCLUDE_DIRS})
   if (HDF5_IS_PARALLEL)   # this should be required - needs a better check
        Message(STATUS "Parallel HDF5 supported")
   else()
        Message(STATUS "No parallel HDF5 Support")
   endif()
endif()

# check for FFTW libraries. Cray implementation are not found but have the environmental variable FFTW_ROOT defined.
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{FFTW_ROOT}/lib/pkgconfig/")
pkg_check_modules(FFTW QUIET fftw3 IMPORTED_TARGET)
if (TARGET PkgConfig::FFTW)
    message(STATUS "FFTW found")
    list(APPEND _libraries PkgConfig::FFTW)
    link_libraries(PkgConfig::FFTW)
else()
        message(STATUS "FFTW not found")
        message(STATUS "Excluding FFTW specific source code")
endif()


if(USE_DPI)
   set(SRC_DPI src/Main/RegPlugin.cpp src/Core/DiagnosticHook.cpp src/Core/DiagnosticHookBeam.cpp src/Core/DiagnosticHookLI.cpp)
endif()

if(USE_CUDA)
   set(SRC_CUDA src/Core/FieldSolverADICuda.cpp src/Core/FieldSolverFFTCuda.cpp src/Cuda/GenesisCudaKernels.cu)
   list(APPEND _libraries ${GENESIS_CUDA_RUNTIME_LIBRARY} ${GENESIS_CUDA_CUFFT_LIBRARY})
endif()


add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
    ALL
    COMMAND
    ${CMAKE_COMMAND} -D TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/git-hash.cmake
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    get_git_hash
    ALL
    DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
   
)


include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# some status print out before compilation
message(STATUS "Build Info:")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )
MESSAGE(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
Message(STATUS "Used libraries: ${_libraries}")

# objects for the main part - maybe needs a better name
add_library(genesis13 STATIC
        src/Util/BesselJ.cpp
        src/Util/GaussHermite.cpp
        src/Util/Hammerslay.cpp
        src/Util/Inverfc.cpp
        src/Util/RandomU.cpp
        src/Util/GenesisNvtx.cpp
	    src/Util/SimpleHandshake.cpp
        src/Util/Sorting.cpp
        src/Util/StringProcessing.cpp
        src/Util/SemaFile.cpp
        src/Util/FFTObj.cpp
        src/Lattice/AlterLattice.cpp
        src/Lattice/Lattice.cpp
        src/Lattice/LatticeElements.cpp
        src/Lattice/LatticeParser.cpp
        src/Lattice/Optics.cpp
        src/Core/Beam.cpp
        src/Core/BeamSolver.cpp
        src/Core/Collective.cpp
        src/Core/Control.cpp
        src/Core/EFieldSolver.cpp
        src/Core/Field.cpp
        src/Core/FieldSolverADI.cpp
        src/Core/FieldSolverFFT.cpp
        src/Core/Gencore.cpp
        src/Core/Incoherent.cpp
        src/Core/TrackBeam.cpp
        src/Core/Undulator.cpp
        src/Core/Diagnostic.cpp
        src/Core/DiagnosticUser.cpp
        src/IO/HDF5base.cpp
        src/IO/Output.cpp
        src/IO/readBeamHDF5.cpp
        src/IO/readFieldHDF5.cpp
        src/IO/SDDSBeam.cpp
        src/IO/writeBeamHDF5.cpp
        src/IO/writeFieldHDF5.cpp
        src/IO/writeWakeHDF5.cpp
        src/IO/readMapHDF5.cpp
        src/Loading/ImportBeam.cpp
        src/Loading/ImportField.cpp
        src/Loading/LoadBeam.cpp
        src/Loading/LoadField.cpp
        src/Loading/GenProfile.cpp
        src/Loading/QuietLoading.cpp
        src/Loading/Series.cpp
        src/Loading/ShotNoise.cpp
        src/Loading/SeriesManager.cpp
        src/Loading/SeriesParser.cpp
        src/Loading/Sequence.cpp
        src/Loading/ImportTransformation.cpp
        src/Main/AlterSetup.cpp
        src/Main/Dump.cpp
        src/Main/AlterBeam.cpp
        src/Main/EField.cpp
        src/Main/FieldManipulator.cpp
        src/Main/GenMain.cpp
        src/Main/mainwrap.cpp
        src/Main/Parser.cpp
        src/Main/Setup.cpp
        src/Main/SponRad.cpp
        src/Main/GenTime.cpp
        src/Main/Track.cpp
        src/Main/Wake.cpp
        ${SRC_DPI}
        ${SRC_CUDA}
)



target_include_directories(genesis13
	PRIVATE
	${CMAKE_CURRENT_BINARY_DIR}/include
)


# add optional macro definitions
if (TARGET PkgConfig::FFTW)
    message(STATUS "Defining the macro FFTW in the source code")
    target_compile_definitions(genesis13 PRIVATE FFTW=1)
endif()

if(USE_CUDA)
   target_compile_definitions(genesis13 PRIVATE GENESIS_CUDA=1)
   target_include_directories(genesis13 PRIVATE ${GENESIS_CUDA_INCLUDE_DIRS})
   if(GENESIS_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
      set_property(TARGET genesis13 PROPERTY CUDA_ARCHITECTURES ${GENESIS_CUDA_ARCHITECTURES})
   endif()
endif()


if(GENESIS_ENABLE_NVTX)
   target_compile_definitions(genesis13 PRIVATE GENESIS_NVTX=1)
endif()

if(USE_DPI)
   target_compile_definitions(genesis13 PRIVATE USE_DPI)
endif()


add_dependencies(genesis13 get_git_hash)

add_executable(genesis4 src/Main/mainwrap.cpp)

if(USE_CUDA)
   target_compile_definitions(genesis4 PRIVATE GENESIS_CUDA=1)
   target_include_directories(genesis4 PRIVATE ${GENESIS_CUDA_INCLUDE_DIRS})
endif()

if(GENESIS_ENABLE_NVTX)
   target_compile_definitions(genesis4 PRIVATE GENESIS_NVTX=1)
endif()

target_link_libraries(genesis4 genesis13 stdc++ ${_libraries})


# Compilation of plugin testbench only when plugin system is requested
#if(USE_DPI)
#   add_subdirectory(misc/plugin_testbench)
#endif()
