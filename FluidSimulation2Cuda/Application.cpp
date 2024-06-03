#include "Application.hpp"
#include "Timer.hpp"

Environment* globalEnvironment;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Check for specific key presses and actions
	if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		// Call your method for W key press (e.g., move object up)
		//yourObject.moveUp();
		//printf("W key pressed\n");
		globalEnvironment->moveUp();
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		// Call your method for S key press (e.g., move object down)
		//printf("S key pressed\n");
		globalEnvironment->moveDown();
	}
	else if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		// Call your method for A key press (e.g., move object left)
		//printf("A key pressed\n");
		globalEnvironment->moveLeft();
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		// Call your method for D key press (e.g., move object right)
		//printf("D key pressed\n");
		globalEnvironment->moveRight();
	}
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		globalEnvironment->turnGenerators();
	}
	else if (key == GLFW_KEY_T && action == GLFW_PRESS) {
		globalEnvironment->turnSurfaceTension();
	}
	else if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
		globalEnvironment->deleteLastObstacle();
	}
	// ... (similar checks for other keys as needed)
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

Environment* initializeEnvironment0(int screenWidth, int screenHeight) {

	int particleCount = 6000;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	Surface2D spawnArea;

	std::vector<ConsumerPipe> consumers;
	std::vector<GeneratorPipe> generators;

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Environment* initializeEnvironment1(int screenWidth, int screenHeight) {

	int particleCount = 6000;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	obstacles.push_back(Surface2D(screenWidth * 2 / 4, 0, screenWidth * 3 / 4, 0));
	obstacles.push_back(Surface2D(screenWidth * 3 / 4, 0, screenWidth * 3 / 4, screenHeight * 3 / 4));
	obstacles.push_back(Surface2D(screenWidth * 3 / 4, screenHeight * 3 / 4, screenWidth * 2 / 4, screenHeight * 3 / 4));
	obstacles.push_back(Surface2D(screenWidth * 2 / 4, screenHeight * 3 / 4, screenWidth * 2 / 4, 0));

	Surface2D spawnArea;

	std::vector<ConsumerPipe> consumers;
	std::vector<GeneratorPipe> generators;

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Environment* initializeEnvironment2(int screenWidth, int screenHeight) {

	int particleCount = 6000;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	obstacles.push_back(Surface2D(screenWidth * 2 / 4, 0, screenWidth * 3 / 4, 0));
	obstacles.push_back(Surface2D(screenWidth * 3 / 4, 0, screenWidth * 3 / 4, screenHeight * 3 / 4));
	obstacles.push_back(Surface2D(screenWidth * 3 / 4, screenHeight * 3 / 4, screenWidth * 2 / 4, screenHeight * 3 / 4));
	obstacles.push_back(Surface2D(screenWidth * 2 / 4, screenHeight * 3 / 4, screenWidth * 2 / 4, 0));

	Surface2D spawnArea;

	std::vector<ConsumerPipe> consumers;

	consumers.push_back(ConsumerPipe(Vector2D(screenWidth - 100, screenHeight - 100), 2));

	std::vector<GeneratorPipe> generators;

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Environment* initializeEnvironment5(int screenWidth, int screenHeight) {

	int particleCount = 5000;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	/*obstacles.push_back(Surface2D(800, 200, 1000, 200));
	obstacles.push_back(Surface2D(1000, 200, 1000, 600));
	obstacles.push_back(Surface2D(1000, 600, 800, 600));
	obstacles.push_back(Surface2D(800, 600, 800, 200));*/

	obstacles.push_back(Surface2D(screenWidth - 150, 0, screenWidth - 150, 200));
	obstacles.push_back(Surface2D(screenWidth - 150, 0, screenWidth - 150, 200));


	obstacles.push_back(Surface2D(screenWidth - 150, 0, screenWidth - 150, 200));
	obstacles.push_back(Surface2D(screenWidth - 50, 0, screenWidth - 50, 200));

	Surface2D spawnArea;

	std::vector<ConsumerPipe> consumers;

	//consumers.push_back(ConsumerPipe(Vector2D(100, screenHeight - 100), 2));

	std::vector<GeneratorPipe> generators;

	generators.push_back(GeneratorPipe(Vector2D(screenWidth - 100, 100), 2));

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Environment* initializeEnvironment10(int screenWidth, int screenHeight) {

	int particleCount = 2500;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	obstacles.push_back(Surface2D(400, 0, 500, 0));
	obstacles.push_back(Surface2D(500, 0, 500, screenHeight));
	obstacles.push_back(Surface2D(500, screenHeight, 400, screenHeight));
	obstacles.push_back(Surface2D(400, screenHeight, 400, 0));

	Surface2D spawnArea = Surface2D(Vector2D(100, 100), Vector2D(300, screenHeight - 100));

	std::vector<ConsumerPipe> consumers;

	std::vector<GeneratorPipe> generators;

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Environment* initializeEnvironment11(int screenWidth, int screenHeight) {

	int particleCount = 2500;

	int particleRadius = 2;
	int particleRadiusOfRepel = 50;

	float particleRepulsionForce = 1.0f;

	float viscosityStrength = 0.1f;

	float how_far_into_the_future = 10.0f;

	int thread_count = 4;

	int interactionMatrixRows = screenHeight / particleRadiusOfRepel;
	int interactionMatrixCols = screenWidth / particleRadiusOfRepel;

	Environment* environment;

	std::vector<Surface2D> obstacles;

	obstacles.push_back(Surface2D(400, 0, 500, 0));
	obstacles.push_back(Surface2D(500, 0, 500, screenHeight / 3));
	obstacles.push_back(Surface2D(500, screenHeight / 3, 400, screenHeight / 3));
	obstacles.push_back(Surface2D(400, screenHeight / 3, 400, 0));

	obstacles.push_back(Surface2D(400, 2 * screenHeight / 3, 500, 2 * screenHeight / 3));
	obstacles.push_back(Surface2D(500, 2 * screenHeight / 3, 500, screenHeight));
	obstacles.push_back(Surface2D(500, screenHeight, 400, screenHeight));
	obstacles.push_back(Surface2D(400, screenHeight, 400, 2 * screenHeight / 3));

	obstacles.push_back(Surface2D(400, screenHeight / 3, 500, screenHeight / 3));
	obstacles.push_back(Surface2D(500, screenHeight / 3, 500, 2 * screenHeight / 3));
	obstacles.push_back(Surface2D(500, 2 * screenHeight / 3, 400, 2 * screenHeight / 3));
	obstacles.push_back(Surface2D(400, 2 * screenHeight / 3, 400, screenHeight / 3));

	Surface2D spawnArea = Surface2D(Vector2D(100, 100), Vector2D(300, screenHeight - 100));

	std::vector<ConsumerPipe> consumers;

	std::vector<GeneratorPipe> generators;

	//environment = new Environment(obstacles, consumers, generators);

	environment = new Environment(particleCount, particleRadius, particleRadiusOfRepel, particleRepulsionForce, screenWidth,
		screenHeight, viscosityStrength, how_far_into_the_future, thread_count,
		interactionMatrixRows, interactionMatrixCols, obstacles, consumers, generators, spawnArea);

	return environment;
}

Application::Application()
{
	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return;
	}

	int screenWidth = 1800;
	int screenHeight = 900;

	// Create a windowed mode window and its OpenGL context
	m_window = glfwCreateWindow(screenWidth, screenHeight, "Circle Example", NULL, NULL);
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

	// Set the key callback function
	glfwSetKeyCallback(m_window, keyCallback);

	m_environment = initializeEnvironment0(screenWidth, screenHeight);
	globalEnvironment = m_environment;
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
