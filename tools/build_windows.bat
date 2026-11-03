@echo off
rem BioOptoMD – editierbarer Build unter Windows ohne Conda (MSVC + CUDA 12.x + Ninja).
rem Aufruf aus dem Repository-Wurzelverzeichnis:  tools\build_windows.bat [Release^|Debug]
rem Voraussetzungen: VS Build Tools 2022 (C++), CUDA Toolkit 12.x, Python-venv in .venv mit
rem   pip install numpy scipy pytest ruff nanobind scikit-build-core ninja cupy-cuda12x
rem CUDA-Version über BOMD_CUDA_PATH überschreibbar (Standard: v12.9).
setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSINSTALL=%%i
if not defined VSINSTALL (echo MSVC nicht gefunden & exit /b 1)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if not defined BOMD_CUDA_PATH set "BOMD_CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%BOMD_CUDA_PATH%"
set "CUDACXX=%BOMD_CUDA_PATH%\bin\nvcc.exe"
set "PATH=%BOMD_CUDA_PATH%\bin;%PATH%"
set CMAKE_GENERATOR=Ninja

.venv\Scripts\python -m pip install -v -e . --no-build-isolation -Ccmake.build-type=%BUILD_TYPE%
