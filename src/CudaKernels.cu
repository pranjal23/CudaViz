#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdio>

// Simple test kernel
__global__ void testKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Create a simple pattern to verify CUDA is working
    output[y * width + x] = (float)(x + y) / (width + height);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mandelbrot Set Kernel
__global__ void mandelbrotKernel(float* output, int width, int height, float zoom, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float real = (x - width / 2.0f) / (zoom * width / 4.0f) + offsetX;
    float imag = (y - height / 2.0f) / (zoom * height / 4.0f) + offsetY;
    
    float zReal = 0.0f;
    float zImag = 0.0f;
    int iterations = 0;
    const int maxIterations = 100;
    
    while (zReal * zReal + zImag * zImag < 4.0f && iterations < maxIterations) {
        float newReal = zReal * zReal - zImag * zImag + real;
        zImag = 2.0f * zReal * zImag + imag;
        zReal = newReal;
        iterations++;
    }
    
    output[y * width + x] = (float)iterations / maxIterations;
}

// Julia Set Kernel
__global__ void juliaKernel(float* output, int width, int height, float cReal, float cImag, float zoom) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float real = (x - width / 2.0f) / (zoom * width / 4.0f);
    float imag = (y - height / 2.0f) / (zoom * height / 4.0f);
    
    float zReal = real;
    float zImag = imag;
    int iterations = 0;
    const int maxIterations = 100;
    
    while (zReal * zReal + zImag * zImag < 4.0f && iterations < maxIterations) {
        float newReal = zReal * zReal - zImag * zImag + cReal;
        zImag = 2.0f * zReal * zImag + cImag;
        zReal = newReal;
        iterations++;
    }
    
    output[y * width + x] = (float)iterations / maxIterations;
}

// Sine Wave Kernel
__global__ void sineWaveKernel(float* output, int width, int height, float frequency, float amplitude, float phase) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / width;
    float v = (float)y / height;
    
    float value = amplitude * sinf(frequency * 2.0f * M_PI * u + phase) * sinf(frequency * 2.0f * M_PI * v + phase);
    output[y * width + x] = (value + amplitude) / (2.0f * amplitude);
}

// Heat Equation Kernel
__global__ void heatEquationKernel(float* output, int width, int height, float time, float diffusion) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / width - 0.5f;
    float v = (float)y / height - 0.5f;
    
    float r2 = u * u + v * v;
    float value = expf(-r2 / (4.0f * diffusion * time + 0.01f)) / (4.0f * M_PI * diffusion * time + 0.01f);
    
    output[y * width + x] = value;
}

// Game of Life Kernel
__global__ void gameOfLifeKernel(float* output, float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            
            if (input[ny * width + nx] > 0.5f) {
                neighbors++;
            }
        }
    }
    
    bool alive = input[y * width + x] > 0.5f;
    bool newAlive = (alive && (neighbors == 2 || neighbors == 3)) || (!alive && neighbors == 3);
    
    output[y * width + x] = newAlive ? 1.0f : 0.0f;
}

// Initialize Game of Life
__global__ void initGameOfLifeKernel(float* output, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    curandState state;
    curand_init(seed + y * width + x, 0, 0, &state);
    
    output[y * width + x] = curand_uniform(&state) > 0.7f ? 1.0f : 0.0f;
}

// Perlin Noise Helper Functions
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float grad(int hash, float x, float y) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : 0;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Perlin Noise Kernel
__global__ void perlinNoiseKernel(float* output, int width, int height, float scale, int octaves) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / width * scale;
    float v = (float)y / height * scale;
    
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        float sampleX = u * frequency;
        float sampleY = v * frequency;
        
        int X = (int)floorf(sampleX) & 255;
        int Y = (int)floorf(sampleY) & 255;
        
        sampleX -= floorf(sampleX);
        sampleY -= floorf(sampleY);
        
        float u_fade = fade(sampleX);
        float v_fade = fade(sampleY);
        
        // Simple hash function for demo
        int p[512];
        for (int j = 0; j < 256; j++) {
            p[j] = j;
            p[256 + j] = j;
        }
        
        int A = p[X] + Y;
        int B = p[X + 1] + Y;
        
        float noise = lerp(
            lerp(grad(p[A], sampleX, sampleY), grad(p[B], sampleX - 1, sampleY), u_fade),
            lerp(grad(p[A + 1], sampleX, sampleY - 1), grad(p[B + 1], sampleX - 1, sampleY - 1), u_fade),
            v_fade
        );
        
        value += noise * amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    
    output[y * width + x] = (value + 1.0f) * 0.5f;
}

// Wave Interference Kernel
__global__ void waveInterferenceKernel(float* output, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / width - 0.5f;
    float v = (float)y / height - 0.5f;
    
    // Multiple wave sources
    float wave1 = sinf(sqrtf((u - 0.2f) * (u - 0.2f) + (v - 0.2f) * (v - 0.2f)) * 30.0f - time * 5.0f);
    float wave2 = sinf(sqrtf((u + 0.2f) * (u + 0.2f) + (v - 0.2f) * (v - 0.2f)) * 25.0f - time * 4.0f);
    float wave3 = sinf(sqrtf((u) * (u) + (v + 0.3f) * (v + 0.3f)) * 35.0f - time * 6.0f);
    
    float value = (wave1 + wave2 + wave3) / 3.0f;
    output[y * width + x] = (value + 1.0f) * 0.5f;
}

// Fluid Simulation Kernel (simplified)
__global__ void fluidSimulationKernel(float* output, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / width;
    float v = (float)y / height;
    
    // Simulate fluid flow with sine waves
    float flowX = sinf(u * 4.0f + time) * cosf(v * 3.0f + time * 0.5f);
    float flowY = cosf(u * 3.0f + time * 0.7f) * sinf(v * 4.0f + time);
    
    float density = 0.5f + 0.3f * sinf(flowX * 10.0f + flowY * 8.0f + time * 2.0f);
    
    output[y * width + x] = fmaxf(0.0f, fminf(1.0f, density));
}

// Kernel launch wrappers
extern "C" {
    void launchTestKernel(float* output, int width, int height) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        testKernel<<<gridSize, blockSize>>>(output, width, height);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA test kernel launch failed: %s\n", cudaGetErrorString(error));
        }
    }

    void launchMandelbrotKernel(float* output, int width, int height, float zoom, float offsetX, float offsetY) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        mandelbrotKernel<<<gridSize, blockSize>>>(output, width, height, zoom, offsetX, offsetY);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
        }
    }
    
    void launchJuliaKernel(float* output, int width, int height, float cReal, float cImag, float zoom) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        juliaKernel<<<gridSize, blockSize>>>(output, width, height, cReal, cImag, zoom);
    }
    
    void launchSineWaveKernel(float* output, int width, int height, float frequency, float amplitude, float phase) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        sineWaveKernel<<<gridSize, blockSize>>>(output, width, height, frequency, amplitude, phase);
    }
    
    void launchHeatEquationKernel(float* output, int width, int height, float time, float diffusion) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        heatEquationKernel<<<gridSize, blockSize>>>(output, width, height, time, diffusion);
    }
    
    void launchGameOfLifeKernel(float* output, float* input, int width, int height) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        gameOfLifeKernel<<<gridSize, blockSize>>>(output, input, width, height);
    }
    
    void launchInitGameOfLifeKernel(float* output, int width, int height, unsigned int seed) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        initGameOfLifeKernel<<<gridSize, blockSize>>>(output, width, height, seed);
    }
    
    void launchPerlinNoiseKernel(float* output, int width, int height, float scale, int octaves) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        perlinNoiseKernel<<<gridSize, blockSize>>>(output, width, height, scale, octaves);
    }
    
    void launchWaveInterferenceKernel(float* output, int width, int height, float time) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        waveInterferenceKernel<<<gridSize, blockSize>>>(output, width, height, time);
    }
    
    void launchFluidSimulationKernel(float* output, int width, int height, float time) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        fluidSimulationKernel<<<gridSize, blockSize>>>(output, width, height, time);
    }
}
