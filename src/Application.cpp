#include "Application.h"
#include <iostream>
#include <algorithm>
#include <GLFW/glfw3.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

Application::Application() 
    : m_currentMode(VisualizationMode::MANDELBROT)
    , m_startTime(std::chrono::high_resolution_clock::now()) {
}

Application::~Application() {
    cleanup();
}

bool Application::initialize() {
    // Create window
    m_window = std::make_unique<Window>(1200, 800, "CUDA + Vulkan Function Visualizer");
    if (!m_window->initialize()) {
        std::cerr << "Failed to initialize window" << std::endl;
        return false;
    }
    
    // Set up input callbacks
    m_window->setKeyCallback([this](int key, int scancode, int action, int mods) {
        onKey(key, scancode, action, mods);
    });
    m_window->setMouseButtonCallback([this](int button, int action, int mods) {
        onMouseButton(button, action, mods);
    });
    m_window->setCursorPosCallback([this](double xpos, double ypos) {
        onCursorPos(xpos, ypos);
    });
    m_window->setScrollCallback([this](double xoffset, double yoffset) {
        onScroll(xoffset, yoffset);
    });
    
    // Initialize Vulkan renderer
    m_renderer = std::make_unique<VulkanRenderer>();
    if (!m_renderer->initialize(m_window->getWindow())) {
        std::cerr << "Failed to initialize Vulkan renderer" << std::endl;
        return false;
    }
    
    // Initialize CUDA compute
    m_cudaCompute = std::make_unique<CudaCompute>();
    if (!m_cudaCompute->initialize()) {
        std::cerr << "Failed to initialize CUDA compute" << std::endl;
        return false;
    }
    
    // Initial visualization
    updateVisualization();
    
    std::cout << "CUDA + Vulkan Function Visualizer initialized successfully!" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  1-8: Switch visualization modes" << std::endl;
    std::cout << "  Mouse: Pan (drag) and Zoom (scroll)" << std::endl;
    std::cout << "  Arrow keys: Adjust parameters" << std::endl;
    std::cout << "  Space: Reset parameters" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    
    return true;
}

void Application::run() {
    while (!m_window->shouldClose()) {
        m_window->pollEvents();
        handleInput();
        update();
        render();
    }
    
    m_renderer->waitIdle();
}

void Application::cleanup() {
    if (m_renderer) {
        m_renderer->cleanup();
    }
    if (m_cudaCompute) {
        m_cudaCompute->cleanup();
    }
    if (m_window) {
        m_window->cleanup();
    }
}

void Application::handleInput() {
    // Additional input handling can be added here
}

void Application::update() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_startTime).count();
    
    // Update time-dependent parameters
    m_params.time = time;
    m_params.phase = time * 2.0f;
    
    // Update visualizations that depend on time
    if (m_currentMode == VisualizationMode::SINE_WAVE ||
        m_currentMode == VisualizationMode::HEAT_EQUATION ||
        m_currentMode == VisualizationMode::WAVE_INTERFERENCE ||
        m_currentMode == VisualizationMode::FLUID_SIMULATION) {
        updateVisualization();
    }
}

void Application::render() {
    m_renderer->render();
}

void Application::updateVisualization() {
    ComputeResult result;
    
    switch (m_currentMode) {
        case VisualizationMode::MANDELBROT:
            result = m_cudaCompute->mandelbrotSet(TEXTURE_WIDTH, TEXTURE_HEIGHT, 
                                                m_params.zoom, m_params.offsetX, m_params.offsetY);
            break;
        case VisualizationMode::JULIA:
            result = m_cudaCompute->juliaSet(TEXTURE_WIDTH, TEXTURE_HEIGHT, 
                                           m_params.cReal, m_params.cImag, m_params.zoom);
            break;
        case VisualizationMode::SINE_WAVE:
            result = m_cudaCompute->sineWave(TEXTURE_WIDTH, TEXTURE_HEIGHT, 
                                           m_params.frequency, m_params.amplitude, m_params.phase);
            break;
        case VisualizationMode::HEAT_EQUATION:
            result = m_cudaCompute->heatEquation(TEXTURE_WIDTH, TEXTURE_HEIGHT, 
                                               m_params.time, m_params.diffusion);
            break;
        case VisualizationMode::GAME_OF_LIFE:
            result = m_cudaCompute->gameOfLife(TEXTURE_WIDTH, TEXTURE_HEIGHT, m_params.generations);
            break;
        case VisualizationMode::PERLIN_NOISE:
            result = m_cudaCompute->perlinNoise(TEXTURE_WIDTH, TEXTURE_HEIGHT, 
                                              m_params.scale, m_params.octaves);
            break;
        case VisualizationMode::WAVE_INTERFERENCE:
            result = m_cudaCompute->waveInterference(TEXTURE_WIDTH, TEXTURE_HEIGHT, m_params.time);
            break;
        case VisualizationMode::FLUID_SIMULATION:
            result = m_cudaCompute->fluidSimulation(TEXTURE_WIDTH, TEXTURE_HEIGHT, m_params.time);
            break;
    }
    
    m_renderer->updateTexture(result.data, result.width, result.height, 
                             result.minValue, result.maxValue);
}

void Application::onKey(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        bool needsUpdate = false;
        
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(m_window->getWindow(), GLFW_TRUE);
                break;
                
            case GLFW_KEY_1:
                m_currentMode = VisualizationMode::MANDELBROT;
                needsUpdate = true;
                break;
            case GLFW_KEY_2:
                m_currentMode = VisualizationMode::JULIA;
                needsUpdate = true;
                break;
            case GLFW_KEY_3:
                m_currentMode = VisualizationMode::SINE_WAVE;
                needsUpdate = true;
                break;
            case GLFW_KEY_4:
                m_currentMode = VisualizationMode::HEAT_EQUATION;
                needsUpdate = true;
                break;
            case GLFW_KEY_5:
                m_currentMode = VisualizationMode::GAME_OF_LIFE;
                needsUpdate = true;
                break;
            case GLFW_KEY_6:
                m_currentMode = VisualizationMode::PERLIN_NOISE;
                needsUpdate = true;
                break;
            case GLFW_KEY_7:
                m_currentMode = VisualizationMode::WAVE_INTERFERENCE;
                needsUpdate = true;
                break;
            case GLFW_KEY_8:
                m_currentMode = VisualizationMode::FLUID_SIMULATION;
                needsUpdate = true;
                break;
                
            case GLFW_KEY_SPACE:
                // Reset parameters
                m_params.zoom = 1.0f;
                m_params.offsetX = 0.0f;
                m_params.offsetY = 0.0f;
                m_params.frequency = 1.0f;
                m_params.amplitude = 1.0f;
                m_params.cReal = -0.7f;
                m_params.cImag = 0.27015f;
                m_params.diffusion = 0.1f;
                m_params.generations = 0;
                m_params.scale = 10.0f;
                m_params.octaves = 4;
                needsUpdate = true;
                break;
                
            case GLFW_KEY_UP:
                if (m_currentMode == VisualizationMode::SINE_WAVE) {
                    m_params.frequency += 0.1f;
                } else if (m_currentMode == VisualizationMode::JULIA) {
                    m_params.cImag += 0.01f;
                } else if (m_currentMode == VisualizationMode::HEAT_EQUATION) {
                    m_params.diffusion += 0.01f;
                } else if (m_currentMode == VisualizationMode::PERLIN_NOISE) {
                    m_params.scale += 1.0f;
                }
                needsUpdate = true;
                break;
                
            case GLFW_KEY_DOWN:
                if (m_currentMode == VisualizationMode::SINE_WAVE) {
                    m_params.frequency = std::max(0.1f, m_params.frequency - 0.1f);
                } else if (m_currentMode == VisualizationMode::JULIA) {
                    m_params.cImag -= 0.01f;
                } else if (m_currentMode == VisualizationMode::HEAT_EQUATION) {
                    m_params.diffusion = std::max(0.01f, m_params.diffusion - 0.01f);
                } else if (m_currentMode == VisualizationMode::PERLIN_NOISE) {
                    m_params.scale = std::max(1.0f, m_params.scale - 1.0f);
                }
                needsUpdate = true;
                break;
                
            case GLFW_KEY_LEFT:
                if (m_currentMode == VisualizationMode::JULIA) {
                    m_params.cReal -= 0.01f;
                } else if (m_currentMode == VisualizationMode::SINE_WAVE) {
                    m_params.amplitude = std::max(0.1f, m_params.amplitude - 0.1f);
                } else if (m_currentMode == VisualizationMode::PERLIN_NOISE) {
                    m_params.octaves = std::max(1, m_params.octaves - 1);
                }
                needsUpdate = true;
                break;
                
            case GLFW_KEY_RIGHT:
                if (m_currentMode == VisualizationMode::JULIA) {
                    m_params.cReal += 0.01f;
                } else if (m_currentMode == VisualizationMode::SINE_WAVE) {
                    m_params.amplitude += 0.1f;
                } else if (m_currentMode == VisualizationMode::PERLIN_NOISE) {
                    m_params.octaves = std::min(8, m_params.octaves + 1);
                }
                needsUpdate = true;
                break;
        }
        
        if (needsUpdate) {
            updateVisualization();
        }
    }
}

void Application::onMouseButton(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            m_dragging = true;
            double xpos, ypos;
            glfwGetCursorPos(m_window->getWindow(), &xpos, &ypos);
            m_lastMouseX = xpos;
            m_lastMouseY = ypos;
        } else if (action == GLFW_RELEASE) {
            m_dragging = false;
        }
    }
}

void Application::onCursorPos(double xpos, double ypos) {
    if (m_dragging) {
        double deltaX = xpos - m_lastMouseX;
        double deltaY = ypos - m_lastMouseY;
        
        // Pan the view
        m_params.offsetX -= static_cast<float>(deltaX * 0.001f / m_params.zoom);
        m_params.offsetY += static_cast<float>(deltaY * 0.001f / m_params.zoom);
        
        m_lastMouseX = xpos;
        m_lastMouseY = ypos;
        
        updateVisualization();
    }
}

void Application::onScroll(double xoffset, double yoffset) {
    // Zoom
    float zoomFactor = 1.1f;
    if (yoffset > 0) {
        m_params.zoom *= zoomFactor;
    } else {
        m_params.zoom /= zoomFactor;
    }
    
    updateVisualization();
}
