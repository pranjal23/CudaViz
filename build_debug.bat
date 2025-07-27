@echo off
echo Building CUDA + Vulkan Function Visualizer (Debug)...

REM Check if build_debug directory exists
if not exist "build_debug" (
    echo Creating build_debug directory...
    mkdir build_debug
)

cd build_debug

REM Configure with CMake (no need for CMAKE_BUILD_TYPE with Visual Studio generator)
echo Configuring project for Debug...
cmake ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project in Debug configuration
echo Building project in Debug configuration...
cmake --build . --config Debug

if %ERRORLEVEL% neq 0 (
    echo Debug build failed!
    pause
    exit /b 1
)

echo Debug build completed successfully!
echo.
echo Available debug executables:
if exist "Debug\CudaVisualizerConsole.exe" (
    echo   Console version: .\Debug\CudaVisualizerConsole.exe
)
if exist "Debug\CudaVulkanVisualizer.exe" (
    echo   Full version: .\Debug\CudaVulkanVisualizer.exe
)
echo.
echo Running debug console version...
if exist "Debug\CudaVisualizerConsole.exe" (
    .\Debug\CudaVisualizerConsole.exe
) else (
    echo Debug console version not found. Check build logs above.
)
echo.
pause
