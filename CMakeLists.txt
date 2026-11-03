# BioOptoMD – Build-Entwurf (Phase 0, ungetestet). Phase 1 macht ihn lauffähig.
cmake_minimum_required(VERSION 3.26)
project(biooptomd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)   # RTX 3090, RTX 6000 Ada
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- Kernbibliothek -----------------------------------------------------------
file(GLOB_RECURSE BOMD_SOURCES CONFIGURE_DEPENDS
     src/core/*.cu src/core/*.cpp
     src/forces/*.cu src/forces/*.cpp
     src/nonbonded/*.cu src/nonbonded/*.cpp
     src/observers/*.cu src/observers/*.cpp)

add_library(bomd_core STATIC ${BOMD_SOURCES})
target_include_directories(bomd_core PUBLIC include PRIVATE src)
target_link_libraries(bomd_core PUBLIC CUDA::cudart)
set_target_properties(bomd_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(bomd_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills --expt-relaxed-constexpr -lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)

# --- Python-Modul ---------------------------------------------------------------
nanobind_add_module(_core NB_STATIC src/bindings/module.cpp)
target_link_libraries(_core PRIVATE bomd_core)
install(TARGETS _core LIBRARY DESTINATION biooptomd)
