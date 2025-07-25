@echo off
echo Checking dependencies for CUDA + Vulkan Visualizer...
echo ================================================
echo.

echo CUDA:
echo -----
nvcc --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [X] CUDA not found
    echo    Download from: https://developer.nvidia.com/cuda-downloads
) else (
    echo [OK] CUDA found
)
echo.

echo CMake:
echo ------
cmake --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [X] CMake not found
    echo    Download from: https://cmake.org/download/
) else (
    echo [OK] CMake found
)
echo.

echo Vulkan SDK:
echo -----------
if "%VULKAN_SDK%"=="" (
    echo [X] VULKAN_SDK environment variable not set
) else (
    echo [OK] VULKAN_SDK = %VULKAN_SDK%
)

glslc --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [X] glslc (shader compiler) not found
    echo    Download Vulkan SDK from: https://vulkan.lunarg.com/
) else (
    echo [OK] glslc found
)
echo.

echo GLFW3:
echo ------
if "%GLFW_ROOT%"=="" (
    echo [X] GLFW_ROOT environment variable not set
    echo    Manual installation required
) else (
    echo [OK] GLFW_ROOT = %GLFW_ROOT%
    if exist "%GLFW_ROOT%\lib-vc2022\glfw3.lib" (
        echo [OK] glfw3.lib found
    ) else (
        echo [X] glfw3.lib not found in expected location
    )
)
echo.

echo Build Status:
echo -------------
if exist "build\Release\CudaVisualizerConsole.exe" (
    echo [OK] Console version built
) else (
    echo [X] Console version not built
)

if exist "build\Release\CudaVulkanVisualizer.exe" (
    echo [OK] Full version built
) else (
    echo [X] Full version not built
)
echo.

echo Summary:
echo --------
if exist "build\Release\CudaVisualizerConsole.exe" (
    echo You can run the console version: build\Release\CudaVisualizerConsole.exe
)

echo.
echo For complete installation instructions, see: INSTALLATION_GUIDE.md
echo.
pause
