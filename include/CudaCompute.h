#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

struct ComputeResult {
    std::vector<float> data;
    int width;
    int height;
    float minValue;
    float maxValue;
    std::string functionName;
};

class CudaCompute {
public:
    CudaCompute();
    ~CudaCompute();

    bool initialize();
    void cleanup();

    // Test function
    ComputeResult testKernel(int width, int height);

    // Different CUDA function visualizations
    ComputeResult mandelbrotSet(int width, int height, float zoom, float offsetX, float offsetY);
    ComputeResult juliaSet(int width, int height, float cReal, float cImag, float zoom);
    ComputeResult sineWave(int width, int height, float frequency, float amplitude, float phase);
    ComputeResult heatEquation(int width, int height, float time, float diffusion);
    ComputeResult gameOfLife(int width, int height, int generations);
    ComputeResult perlinNoise(int width, int height, float scale, int octaves);
    ComputeResult waveInterference(int width, int height, float time);
    ComputeResult fluidSimulation(int width, int height, float time);

private:
    bool m_initialized;
    CUcontext m_cudaContext;
    CUdevice m_cudaDevice;
    
    // Helper methods
    void checkCudaError(cudaError_t error, const char* operation);
    void checkCudaDriverError(CUresult result, const char* operation);
};
