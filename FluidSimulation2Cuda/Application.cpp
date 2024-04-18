#include "Application.hpp"
#include "Timer.hpp"



void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

Application::Application()
{
	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return;
	}

	// Create a windowed mode window and its OpenGL context
	m_window = glfwCreateWindow(1280, 720, "Circle Example", NULL, NULL);
	if (!m_window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return;
	}

	// Set framebuffer size callback
	glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

	// Make the window's context current
	glfwMakeContextCurrent(m_window);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	m_environment = new Environment();
}

Application::~Application()
{
	glfwTerminate();
	delete m_environment;
}

void Application::events() {

	// Poll for and process events
	glfwPollEvents();
}

void Application::loop()
{
	float time_passed = 0.0f;
	int frames = 0;
	std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();
	// Loop until the user closes the window
	while (!glfwWindowShouldClose(m_window)) {

		double deltaTime = Timer::getInstance()->getTime();


		//std::cout << deltaTime << std::endl;

		//std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

		this->update(deltaTime);

		std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
		double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

		//std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
		//double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

		//time1 = std::chrono::steady_clock::now();

		time1 = std::chrono::steady_clock::now();

		// Render here
		this->render();

		time2 = std::chrono::steady_clock::now();
		tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

		//time2 = std::chrono::steady_clock::now();
		//tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

		time1 = std::chrono::steady_clock::now();

		this->events();

		time2 = std::chrono::steady_clock::now();
		tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

		std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
		tick = std::chrono::duration_cast<std::chrono::microseconds>(time - lastTime).count() / 1000000.0f;
		lastTime = time;

		time_passed += tick;
		frames++;

		if (time_passed >= 1.0f) {
			std::cout << "FPS: " << frames << " " << std::endl;
			time_passed = 0.0f;
			frames = 0;
		}
	}
}

void Application::mousePress() {
	/*if (b.button == SDL_BUTTON_LEFT) {

	}
	if (b.button)
		if (b.button == SDL_BUTTON_RIGHT) {
			int x = b.x;
			int y = b.y;
		}*/
}

void Application::render()
{
	// Render here
	glClear(GL_COLOR_BUFFER_BIT);

	// Get the size of the window
	glfwGetFramebufferSize(m_window, &m_width, &m_height);

	// Set up the viewport to maintain aspect ratio
	glViewport(0, 0, m_width, m_height);

	// Calculate the aspect ratio
	float aspectRatio = static_cast<float>(m_width) / static_cast<float>(m_height);

	// Apply aspect ratio to your projection matrix or adjust accordingly
	// For example:
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-aspectRatio, aspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);

	// Render environment
	m_environment->render(m_width, m_height);

	// Swap front and back buffers
	glfwSwapBuffers(m_window);
}

void Application::update(float dt)
{
	//m_environment->update(dt);
	m_environment->newUpdate(dt);
}
