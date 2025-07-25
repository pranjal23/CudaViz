#include "CudaCompute.h"
#include <iostream>
#include <algorithm>
#include <random>

// External kernel function declarations
extern "C" {
    void launchTestKernel(float* output, int width, int height);
    void launchMandelbrotKernel(float* output, int width, int height, float zoom, float offsetX, float offsetY);
    void launchJuliaKernel(float* output, int width, int height, float cReal, float cImag, float zoom);
    void launchSineWaveKernel(float* output, int width, int height, float frequency, float amplitude, float phase);
    void launchHeatEquationKernel(float* output, int width, int height, float time, float diffusion);
    void launchGameOfLifeKernel(float* output, float* input, int width, int height);
    void launchInitGameOfLifeKernel(float* output, int width, int height, unsigned int seed);
    void launchPerlinNoiseKernel(float* output, int width, int height, float scale, int octaves);
    void launchWaveInterferenceKernel(float* output, int width, int height, float time);
    void launchFluidSimulationKernel(float* output, int width, int height, float time);
}

CudaCompute::CudaCompute() : m_initialized(false), m_cudaContext(nullptr) {}

CudaCompute::~CudaCompute() {
    cleanup();
}

bool CudaCompute::initialize() {
    try {
        // Initialize CUDA driver API
        CUresult result = cuInit(0);
        checkCudaDriverError(result, "cuInit");
        
        // Get device count
        int deviceCount;
        result = cuDeviceGetCount(&deviceCount);
        checkCudaDriverError(result, "cuDeviceGetCount");
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        // Get first device
        result = cuDeviceGet(&m_cudaDevice, 0);
        checkCudaDriverError(result, "cuDeviceGet");
        
        // Create context
        result = cuCtxCreate(&m_cudaContext, 0, m_cudaDevice);
        checkCudaDriverError(result, "cuCtxCreate");
        
        // Initialize runtime API
        cudaError_t error = cudaSetDevice(0);
        checkCudaError(error, "cudaSetDevice");
        
        m_initialized = true;
        
        // Print device info
        char deviceName[256];
        cuDeviceGetName(deviceName, sizeof(deviceName), m_cudaDevice);
        std::cout << "CUDA initialized successfully on device: " << deviceName << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CudaCompute::cleanup() {
    if (m_initialized) {
        if (m_cudaContext) {
            cuCtxDestroy(m_cudaContext);
            m_cudaContext = nullptr;
        }
        m_initialized = false;
    }
}

ComputeResult CudaCompute::testKernel(int width, int height) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Test Kernel";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for test");
    
    launchTestKernel(d_output, width, height);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for test");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for test");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::mandelbrotSet(int width, int height, float zoom, float offsetX, float offsetY) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Mandelbrot Set";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for mandelbrot");
    
    launchMandelbrotKernel(d_output, width, height, zoom, offsetX, offsetY);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for mandelbrot");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for mandelbrot");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::juliaSet(int width, int height, float cReal, float cImag, float zoom) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Julia Set";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for julia");
    
    launchJuliaKernel(d_output, width, height, cReal, cImag, zoom);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for julia");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for julia");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::sineWave(int width, int height, float frequency, float amplitude, float phase) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Sine Wave";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for sine wave");
    
    launchSineWaveKernel(d_output, width, height, frequency, amplitude, phase);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for sine wave");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for sine wave");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::heatEquation(int width, int height, float time, float diffusion) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Heat Equation";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for heat equation");
    
    launchHeatEquationKernel(d_output, width, height, time, diffusion);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for heat equation");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for heat equation");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

static float* gameOfLifeBuffer1 = nullptr;
static float* gameOfLifeBuffer2 = nullptr;
static int gameOfLifeGenerationCount = 0;

ComputeResult CudaCompute::gameOfLife(int width, int height, int generations) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Game of Life";
    
    size_t dataSize = width * height * sizeof(float);
    
    // Initialize buffers if needed
    if (!gameOfLifeBuffer1 || gameOfLifeGenerationCount == 0) {
        if (gameOfLifeBuffer1) cudaFree(gameOfLifeBuffer1);
        if (gameOfLifeBuffer2) cudaFree(gameOfLifeBuffer2);
        
        cudaError_t error = cudaMalloc(&gameOfLifeBuffer1, dataSize);
        checkCudaError(error, "cudaMalloc for game of life buffer 1");
        
        error = cudaMalloc(&gameOfLifeBuffer2, dataSize);
        checkCudaError(error, "cudaMalloc for game of life buffer 2");
        
        // Initialize with random pattern
        std::random_device rd;
        launchInitGameOfLifeKernel(gameOfLifeBuffer1, width, height, rd());
        gameOfLifeGenerationCount = 0;
    }
    
    // Run simulation for specified generations
    float* currentBuffer = gameOfLifeBuffer1;
    float* nextBuffer = gameOfLifeBuffer2;
    
    for (int i = 0; i < generations; i++) {
        launchGameOfLifeKernel(nextBuffer, currentBuffer, width, height);
        std::swap(currentBuffer, nextBuffer);
        gameOfLifeGenerationCount++;
    }
    
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for game of life");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), currentBuffer, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for game of life");
    
    result.minValue = 0.0f;
    result.maxValue = 1.0f;
    
    return result;
}

ComputeResult CudaCompute::perlinNoise(int width, int height, float scale, int octaves) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Perlin Noise";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for perlin noise");
    
    launchPerlinNoiseKernel(d_output, width, height, scale, octaves);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for perlin noise");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for perlin noise");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::waveInterference(int width, int height, float time) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Wave Interference";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for wave interference");
    
    launchWaveInterferenceKernel(d_output, width, height, time);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for wave interference");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for wave interference");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

ComputeResult CudaCompute::fluidSimulation(int width, int height, float time) {
    ComputeResult result;
    result.width = width;
    result.height = height;
    result.functionName = "Fluid Simulation";
    
    size_t dataSize = width * height * sizeof(float);
    float* d_output;
    
    cudaError_t error = cudaMalloc(&d_output, dataSize);
    checkCudaError(error, "cudaMalloc for fluid simulation");
    
    launchFluidSimulationKernel(d_output, width, height, time);
    
    error = cudaDeviceSynchronize();
    checkCudaError(error, "cudaDeviceSynchronize for fluid simulation");
    
    result.data.resize(width * height);
    error = cudaMemcpy(result.data.data(), d_output, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy for fluid simulation");
    
    cudaFree(d_output);
    
    auto minMax = std::minmax_element(result.data.begin(), result.data.end());
    result.minValue = *minMax.first;
    result.maxValue = *minMax.second;
    
    return result;
}

void CudaCompute::checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::string message = std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(error);
        throw std::runtime_error(message);
    }
}

void CudaCompute::checkCudaDriverError(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::string message = std::string("CUDA driver error in ") + operation + ": " + errorStr;
        throw std::runtime_error(message);
    }
}
