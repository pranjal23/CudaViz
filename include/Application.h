#pragma once

#include "Window.h"
#include "VulkanRenderer.h"
#include "CudaCompute.h"
#include <memory>
#include <chrono>

enum class VisualizationMode {
    MANDELBROT,
    JULIA,
    SINE_WAVE,
    HEAT_EQUATION,
    GAME_OF_LIFE,
    PERLIN_NOISE,
    WAVE_INTERFERENCE,
    FLUID_SIMULATION
};

class Application {
public:
    Application();
    ~Application();

    bool initialize();
    void run();
    void cleanup();

private:
    std::unique_ptr<Window> m_window;
    std::unique_ptr<VulkanRenderer> m_renderer;
    std::unique_ptr<CudaCompute> m_cudaCompute;
    
    VisualizationMode m_currentMode;
    
    // Parameters for different visualizations
    struct {
        float zoom = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float phase = 0.0f;
        float cReal = -0.7f;
        float cImag = 0.27015f;
        float time = 0.0f;
        float diffusion = 0.1f;
        int generations = 0;
        float scale = 10.0f;
        int octaves = 4;
    } m_params;
    
    std::chrono::high_resolution_clock::time_point m_startTime;
    
    void handleInput();
    void update();
    void render();
    void updateVisualization();
    
    // Input callbacks
    void onKey(int key, int scancode, int action, int mods);
    void onMouseButton(int button, int action, int mods);
    void onCursorPos(double xpos, double ypos);
    void onScroll(double xoffset, double yoffset);
    
    // UI state
    bool m_dragging = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    
    const int TEXTURE_WIDTH = 800;
    const int TEXTURE_HEIGHT = 600;
};
