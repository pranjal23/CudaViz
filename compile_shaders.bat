@echo off
echo Compiling Vulkan shaders...

REM Check if glslc is available
where glslc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: glslc not found. Please ensure Vulkan SDK is installed and in PATH.
    pause
    exit /b 1
)

REM Create shaders directory if it doesn't exist
if not exist "shaders" (
    mkdir shaders
)

REM Compile vertex shader
echo Compiling vertex shader...
glslc shaders\vertex.vert -o shaders\vertex.vert.spv
if %ERRORLEVEL% neq 0 (
    echo Error compiling vertex shader!
    pause
    exit /b 1
)

REM Compile fragment shader
echo Compiling fragment shader...
glslc shaders\fragment.frag -o shaders\fragment.frag.spv
if %ERRORLEVEL% neq 0 (
    echo Error compiling fragment shader!
    pause
    exit /b 1
)

echo Shaders compiled successfully!
echo Generated files:
echo   - shaders\vertex.vert.spv
echo   - shaders\fragment.frag.spv
echo.
