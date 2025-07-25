cmake_minimum_required(VERSION 3.18)
project(CudaVulkanVisualizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Vulkan QUIET)
find_package(PkgConfig QUIET)

# Try to find GLFW through different methods
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND AND PkgConfig_FOUND)
    pkg_check_modules(GLFW QUIET glfw3)
endif()

# Try manual path if not found
if(NOT glfw3_FOUND AND NOT GLFW_FOUND)
    if(DEFINED ENV{GLFW_ROOT})
        set(GLFW_ROOT "$ENV{GLFW_ROOT}")
        find_library(GLFW_LIBRARIES
            NAMES glfw3 glfw
            PATHS "${GLFW_ROOT}/lib-vc2022" "${GLFW_ROOT}/lib"
            NO_DEFAULT_PATH
        )
        find_path(GLFW_INCLUDE_DIRS
            NAMES GLFW/glfw3.h
            PATHS "${GLFW_ROOT}/include"
            NO_DEFAULT_PATH
        )
        if(GLFW_LIBRARIES AND GLFW_INCLUDE_DIRS)
            set(GLFW_FOUND TRUE)
            message(STATUS "Found GLFW at: ${GLFW_ROOT}")
        endif()
    endif()
endif()

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61 75 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Check what we have available
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_VERSION}")
    include_directories(${Vulkan_INCLUDE_DIRS})
    set(VULKAN_AVAILABLE TRUE)
else()
    message(WARNING "Vulkan SDK not found. Building without Vulkan support.")
    set(VULKAN_AVAILABLE FALSE)
endif()

if(glfw3_FOUND)
    message(STATUS "GLFW3 found")
    set(GLFW_AVAILABLE TRUE)
elseif(GLFW_FOUND)
    message(STATUS "GLFW3 found via pkg-config")
    set(GLFW_AVAILABLE TRUE)
else()
    message(WARNING "GLFW3 not found. Please install GLFW3.")
    set(GLFW_AVAILABLE FALSE)
endif()

# Source files for console version (CUDA only)
set(CONSOLE_SOURCES
    src/main_console.cpp
    src/CudaCompute.cpp
    src/CudaKernels.cu
)

# Source files for full version (CUDA + Vulkan)
set(FULL_SOURCES
    src/main.cpp
    src/VulkanRenderer.cpp
    src/CudaCompute.cpp
    src/Window.cpp
    src/Application.cpp
    src/CudaKernels.cu
)

# Always build console version
add_executable(CudaVisualizerConsole ${CONSOLE_SOURCES})
target_link_libraries(CudaVisualizerConsole 
    CUDA::cudart
    CUDA::cuda_driver
)
set_property(TARGET CudaVisualizerConsole PROPERTY CUDA_RUNTIME_LIBRARY Static)

# Build full version if Vulkan and GLFW are available
if(VULKAN_AVAILABLE AND GLFW_AVAILABLE)
    add_executable(${PROJECT_NAME} ${FULL_SOURCES})
    
    if(glfw3_FOUND)
        target_link_libraries(${PROJECT_NAME} 
            CUDA::cudart
            CUDA::cuda_driver
            ${Vulkan_LIBRARIES}
            glfw
        )
    else()
        target_link_libraries(${PROJECT_NAME} 
            CUDA::cudart
            CUDA::cuda_driver
            ${Vulkan_LIBRARIES}
            ${GLFW_LIBRARIES}
        )
        target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_INCLUDE_DIRS})
    endif()
    
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Static)
    
    # Copy shaders to build directory if they exist
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    
    message(STATUS "Building full Vulkan version")
else()
    message(STATUS "Building console version only")
    if(NOT VULKAN_AVAILABLE)
        message(STATUS "  - Install Vulkan SDK from: https://vulkan.lunarg.com/")
    endif()
    if(NOT GLFW_AVAILABLE)
        message(STATUS "  - Install GLFW3 library")
    endif()
endif()

# Platform specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()
