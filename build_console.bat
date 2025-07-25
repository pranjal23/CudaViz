@echo off
echo Building CUDA Console Visualizer (No Graphics Dependencies)...
echo =============================================================

REM Check if CUDA is available
nvcc --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: CUDA not found. Please install CUDA Toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

REM Check if CMake is available
cmake --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: CMake not found. Please install CMake.
    echo Download from: https://cmake.org/download/
    pause
    exit /b 1
)

echo ✅ CUDA found
echo ✅ CMake found
echo.

REM Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure project
echo Configuring project for console build...
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CONSOLE_ONLY=ON

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build project
echo Building console version...
cmake --build . --config Release --target CudaVisualizerConsole

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Build completed successfully!
echo.

REM Run the console version
if exist "Release\CudaVisualizerConsole.exe" (
    echo Running CUDA Console Visualizer...
    echo.
    .\Release\CudaVisualizerConsole.exe
) else if exist "CudaVisualizerConsole.exe" (
    echo Running CUDA Console Visualizer...
    echo.
    .\CudaVisualizerConsole.exe
) else (
    echo Console executable not found. Check build output above.
    pause
)

cd ..
