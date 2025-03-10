cmake_minimum_required(VERSION 3.13)
project (DummySPH C CXX)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Since this example uses MPI, find and link against it.
find_package(MPI COMPONENTS C CXX)
if (NOT MPI_FOUND)
  message(STATUS
    "Skipping example: ${CMAKE_PROJECT_NAME} requires MPI.")
  return ()
endif ()

set(C_SRCS
Driver.cxx
)

#add_executable(dummysph ${C_SRCS})
#target_link_libraries(dummysph PRIVATE MPI::MPI_C MPI::MPI_CXX m)
#if(STRIDED_SCALARS)
#  target_compile_definitions(dummysph PRIVATE STRIDED_SCALARS=1)
#endif()
#if(CAN_LOAD_H5Part)
#  target_compile_definitions(dummysph PRIVATE LOAD_H5Part=1)
#endif()

option(STRIDED_SCALARS "Use interleaved scalar arrays") 
option(CAN_LOAD_TIPSY  "Use Tipsy data instead of dummy data")
option(CAN_LOAD_H5Part "Use H5Part data instead of dummy data")

if(CAN_LOAD_H5Part)
find_package(HDF5 REQUIRED)
endif()

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent VTK-m)

if(INSITU STREQUAL "Catalyst")
  find_package(catalyst REQUIRED PATHS "/local/apps/ParaView/")
  # we only need C++ compilers if we're building the Adaptor
  enable_language(CXX)

  add_executable(dummysph_catalystV2 ${C_SRCS})
if(CAN_LOAD_TIPSY)
  target_compile_definitions(dummysph_catalystV2 PRIVATE LOAD_TIPSY=1)
endif()
  target_compile_definitions(dummysph_catalystV2 PRIVATE USE_CATALYST=1)
if(STRIDED_SCALARS)
  target_compile_definitions(dummysph_catalystV2 PRIVATE STRIDED_SCALARS=1)
endif()
  target_link_libraries(dummysph_catalystV2 PRIVATE MPI::MPI_C MPI::MPI_CXX m catalyst::catalyst)

elseif(INSITU STREQUAL "Ascent")
  # find_package(Ascent REQUIRED PATHS "/local/apps/ascent/install/lib/cmake/ascent")
  find_package(Ascent REQUIRED PATHS "/capstor/scratch/cscs/piccinal/daint/jf/ascent/install/ascent-checkout")

  add_executable(dummysph_ascent ${C_SRCS})
if(CAN_LOAD_H5Part)
  target_compile_definitions(dummysph_ascent PRIVATE LOAD_H5Part=1)
endif()
if(CAN_LOAD_TIPSY)
  target_compile_definitions(dummysph_ascent PRIVATE LOAD_TIPSY=1)
endif()
  target_include_directories(dummysph_ascent PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_definitions(dummysph_ascent PRIVATE USE_ASCENT=1)
if(STRIDED_SCALARS)
  target_compile_definitions(dummysph_ascent PRIVATE STRIDED_SCALARS=1)
endif()
  target_link_libraries(dummysph_ascent MPI::MPI_C MPI::MPI_CXX ascent::ascent_mpi)

elseif(INSITU STREQUAL "VTK-m")
#set_source_files_properties(${C_SRCS} PROPERTIES LANGUAGE CUDA)
if(VTKm_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
endif()
if(CUDA_FOUND)
  set_source_files_properties(${C_SRCS} PROPERTIES LANGUAGE CUDA)
endif()
  find_package(VTKm REQUIRED QUIET)

  add_executable(dummysph_vtkm ${C_SRCS})
if(CAN_LOAD_TIPSY)
  target_compile_definitions(dummysph_vtkm PRIVATE LOAD_TIPSY=1)
endif()

  target_compile_definitions(dummysph_vtkm PRIVATE USE_VTKM=1)
if(STRIDED_SCALARS)
  target_compile_definitions(dummysph_vtkm PRIVATE STRIDED_SCALARS=1)
endif()
  target_link_libraries(dummysph_vtkm MPI::MPI_C MPI::MPI_CXX vtkm::io vtkm::filter vtkm::rendering)
endif()

