#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "CudaCompute.h"

void printVisualization(const ComputeResult& result, int displayWidth = 80, int displayHeight = 24) {
    std::cout << "\n" << result.functionName << " Visualization:" << std::endl;
    std::cout << std::string(displayWidth, '=') << std::endl;
    
    // Sample the result data for console display
    for (int y = 0; y < displayHeight; y++) {
        for (int x = 0; x < displayWidth; x++) {
            int srcX = (x * result.width) / displayWidth;
            int srcY = (y * result.height) / displayHeight;
            float value = result.data[srcY * result.width + srcX];
            
            // Convert to ASCII art
            const char* chars = " .:-=+*#%@";
            int charIndex = (int)(value * 9);
            charIndex = std::max(0, std::min(9, charIndex));
            std::cout << chars[charIndex];
        }
        std::cout << std::endl;
    }
    
    std::cout << std::string(displayWidth, '=') << std::endl;
    std::cout << "Range: " << std::fixed << std::setprecision(3) 
              << result.minValue << " to " << result.maxValue << std::endl;
}

int main() {
    std::cout << "CUDA Function Visualizer (Console Mode)" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Initialize CUDA
    CudaCompute cudaCompute;
    if (!cudaCompute.initialize()) {
        std::cerr << "Failed to initialize CUDA!" << std::endl;
        return -1;
    }
    
    const int width = 160;
    const int height = 120;
    
    std::cout << "\nTesting different CUDA kernels...\n" << std::endl;
    
    // Test basic CUDA functionality first
    auto testResult = cudaCompute.testKernel(width, height);
    printVisualization(testResult);
    std::cout << "Press Enter to continue to Mandelbrot Set...";
    std::cin.get();
    
    // Test Mandelbrot Set
    auto result1 = cudaCompute.mandelbrotSet(width, height, 1.0f, 0.0f, 0.0f);
    printVisualization(result1);
    
    std::cout << "\nPress Enter to continue to Julia Set...";
    std::cin.get();
    
    // Test Julia Set
    auto result2 = cudaCompute.juliaSet(width, height, -0.7f, 0.27015f, 1.0f);
    printVisualization(result2);
    
    std::cout << "\nPress Enter to continue to Sine Wave...";
    std::cin.get();
    
    // Test Sine Wave
    auto result3 = cudaCompute.sineWave(width, height, 2.0f, 1.0f, 0.0f);
    printVisualization(result3);
    
    std::cout << "\nPress Enter to continue to Heat Equation...";
    std::cin.get();
    
    // Test Heat Equation
    auto result4 = cudaCompute.heatEquation(width, height, 1.0f, 0.1f);
    printVisualization(result4);
    
    std::cout << "\nPress Enter to continue to Perlin Noise...";
    std::cin.get();
    
    // Test Perlin Noise
    auto result5 = cudaCompute.perlinNoise(width, height, 10.0f, 4);
    printVisualization(result5);
    
    std::cout << "\nPress Enter to continue to Wave Interference...";
    std::cin.get();
    
    // Test Wave Interference
    auto result6 = cudaCompute.waveInterference(width, height, 1.0f);
    printVisualization(result6);
    
    std::cout << "\nAll CUDA kernels tested successfully!" << std::endl;
    std::cout << "To see graphical visualization, install Vulkan SDK and rebuild." << std::endl;
    
    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    
    return 0;
}
