# BioOptoMD – Build (Phase 1). Wird über scikit-build-core aus pyproject.toml aufgerufen.
cmake_minimum_required(VERSION 3.26)

# Muss vor project() stehen: project() setzt sonst den Compiler-Standard (z. B. 52).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)   # RTX 3090, RTX 6000 Ada (AGENTS.md Regel 13)
endif()

project(biooptomd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(NOT arch MATCHES "^(86|89)")
    message(WARNING "CUDA-Architektur ${arch} ist kein Ziel (86;89) – AGENTS.md Regel 13")
  endif()
endforeach()
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- Warnungen (AGENTS.md §6: ohne Warnungen kompilieren) ------------------------------------
if(MSVC)
  set(BOMD_HOST_WARNINGS /W4 /permissive- /utf-8)
else()
  set(BOMD_HOST_WARNINGS -Wall -Wextra)
endif()
string(REPLACE ";" "," BOMD_HOST_WARNINGS_CUDA "${BOMD_HOST_WARNINGS}")

# --- Kernbibliothek -------------------------------------------------------------------------
file(GLOB_RECURSE BOMD_SOURCES CONFIGURE_DEPENDS
     src/core/*.cu src/core/*.cpp
     src/forces/*.cu src/forces/*.cpp
     src/nonbonded/*.cu src/nonbonded/*.cpp
     src/observers/*.cu src/observers/*.cpp)

add_library(bomd_core STATIC ${BOMD_SOURCES})
target_include_directories(bomd_core PUBLIC include PRIVATE src)
target_link_libraries(bomd_core PUBLIC CUDA::cudart_static)
set_target_properties(bomd_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)
# Kein --use_fast_math in Kraft-Kernels (UNITS_AND_CONVENTIONS.md, Numerik).
target_compile_options(bomd_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${BOMD_HOST_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${BOMD_HOST_WARNINGS_CUDA}>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  # Spill-Warnungen nur für optimierten Code: -G erzeugt systematisch Stack-Frames/Spills.
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3 -lineinfo -Xptxas=-warn-spills>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)

# --- Python-Modul ---------------------------------------------------------------------------
nanobind_add_module(_core NB_STATIC src/bindings/module.cpp)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE bomd_core)
target_compile_options(_core PRIVATE ${BOMD_HOST_WARNINGS})
install(TARGETS _core LIBRARY DESTINATION biooptomd)
