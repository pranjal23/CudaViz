@echo off
echo Building CUDA + Vulkan Function Visualizer...

REM Check if build directory exists
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake
echo Configuring project...
cmake .. -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build completed successfully!
echo.
echo Available executables:
if exist "Release\CudaVisualizerConsole.exe" (
    echo   Console version: .\Release\CudaVisualizerConsole.exe
)
if exist "Release\CudaVulkanVisualizer.exe" (
    echo   Full version: .\Release\CudaVulkanVisualizer.exe
)
echo.
echo Running console version...
if exist "Release\CudaVisualizerConsole.exe" (
    .\Release\CudaVisualizerConsole.exe
) else (
    echo Console version not found. Check build logs above.
)
echo.
pause
