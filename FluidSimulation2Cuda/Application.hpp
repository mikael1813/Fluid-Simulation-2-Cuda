#pragma once

//#include <SDL.h>

//#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

#include "Environment.hpp"

class Application
{
public:
    Application();
    ~Application();

    void events();
    void loop();
    void render();

    void update(float dt);

    void mousePress();
private:

    GLFWwindow* m_window;

    Environment* m_environment;

    int m_width, m_height;
};