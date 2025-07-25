#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <functional>

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    bool initialize();
    void cleanup();
    
    bool shouldClose();
    void pollEvents();
    
    GLFWwindow* getWindow() { return m_window; }
    
    void setKeyCallback(std::function<void(int, int, int, int)> callback);
    void setMouseButtonCallback(std::function<void(int, int, int)> callback);
    void setCursorPosCallback(std::function<void(double, double)> callback);
    void setScrollCallback(std::function<void(double, double)> callback);

private:
    GLFWwindow* m_window;
    int m_width;
    int m_height;
    std::string m_title;
    
    std::function<void(int, int, int, int)> m_keyCallback;
    std::function<void(int, int, int)> m_mouseButtonCallback;
    std::function<void(double, double)> m_cursorPosCallback;
    std::function<void(double, double)> m_scrollCallback;
    
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};
