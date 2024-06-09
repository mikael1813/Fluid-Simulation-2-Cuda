#pragma once

//#include <SDL.h>

//#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

#include "Environment.hpp"
#include "Subject.h"


class Application : public Subject
{
public:
	Application(int scenario);
	Application() {}
	~Application();

	void events();
	void loop();
	void render();

	void update(float dt);

	void mousePress();

	void updatePressure(float pressure);
	void updateGravity(float gravity);
	void updateTargetDensity(float targetDensity);
	void updateViscosity(float viscosity);

	//void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
private:

	GLFWwindow* m_window;

	Environment* m_environment;

	int m_width, m_height;
};