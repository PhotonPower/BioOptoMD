@echo off
rem BioOptoMD – baut die GPU-Bibliothek hpfem_gpu.dll von hp-FEM (cuDSS) für den 3D-Optiklöser
rem (ROADMAP 10.5, OQ-26). Aufruf:  tools\build_hpfem_gpu.bat <hp-FEM-Quellen> <Build-Verzeichnis>
rem Voraussetzungen: VS Build Tools 2022, CUDA 12.9, cuDSS 0.8 (lib\12), MSYS2-Ninja als Werkzeug.
rem Danach: HPFEM_GPU_DLL=<Build-Verzeichnis>\bin\hpfem_gpu.dll und das cuDSS-bin\12 auf PATH.
setlocal
if "%~2"=="" (echo Aufruf: build_hpfem_gpu.bat ^<hp-FEM-Quellen^> ^<Build-Verzeichnis^> & exit /b 1)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined BOMD_CUDA_PATH set "BOMD_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%BOMD_CUDA_PATH%"
set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
set "PATH=%CUDA_PATH%\bin;%PATH%"
if not defined BOMD_CUDSS_DIR set "BOMD_CUDSS_DIR=C:\Program Files\NVIDIA cuDSS\v0.8"
cmake -S "%~1\gpu" -B "%~2" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe ^
  -Dcudss_DIR="%BOMD_CUDSS_DIR%\lib\12\cmake\cudss" -DCMAKE_CUDA_ARCHITECTURES=86 || exit /b 1
cmake --build "%~2" || exit /b 1
set "PATH=%BOMD_CUDSS_DIR%\bin\12;%PATH%"
ctest --test-dir "%~2" --output-on-failure
