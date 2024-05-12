#include "Environment.hpp"
#include "Graphics.hpp"
#include "GpuParallel.cuh"


#include <algorithm>

#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <random>

#include <chrono>

#include <mutex> 
#include <map>


//constexpr auto particleCount = 10000;

constexpr auto particleRadius = 2;
constexpr auto particleRadiusOfRepel = 50;
constexpr auto particleDistance = 30;

constexpr auto particleRepulsionForce = 3.0f;

constexpr int SCREEN_WIDTH = 1280;
constexpr int SCREEN_HEIGHT = 720;

//constexpr float viscosityStrength = 0.0f;
constexpr float viscosityStrength = 0.1f;

constexpr float HOW_FAR_INTO_THE_FUTURE = 10.0f;

constexpr int THREAD_COUNT = 4;

int interactionMatrixRows = SCREEN_HEIGHT / particleRadiusOfRepel;
int interactionMatrixCols = SCREEN_WIDTH / particleRadiusOfRepel;

float ExampleFunction(Vector2D point) {
	return cos(point.Y - 3 + sin(point.X));
}


Environment::Environment() {


	InteractionMatrixClass::getInstance()->initializeMatrix(SCREEN_WIDTH, SCREEN_HEIGHT, particleRadiusOfRepel);

	m_Particles = std::vector<Particle>{};

	// Seed the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	int count = 0;

	for (int i = 0; i < m_ParticleCount; i++) {

		bool ok;
		float posX;
		float posY;
		do {
			ok = true;

			posX = std::uniform_int_distribution<int>(100, SCREEN_WIDTH - 100)(gen);
			posY = std::uniform_int_distribution<int>(100, SCREEN_HEIGHT - 100)(gen);

			for (auto& particle : m_Particles) {
				if (Math::squared_distance(particle.getPosition(), Vector2D(posX, posY)) <= particleRadius) {
					ok = false;
					break;
				}
			}
		} while (!ok);

		m_Particles.push_back(Particle(posX, posY, count++));
		m_ParticleProperties.push_back(ExampleFunction(Vector2D(posX, posY)));
		m_ParticleDensities.push_back(0.0f);

	}

	int particleCount = Math::nextPowerOf2(m_Particles.size());

	for (int i = m_Particles.size(); i < particleCount; i++) {
		m_Particles.push_back(Particle());
	}

	m_Obstacles.push_back(Surface2D(50, 10, 1200, 11));
	m_Obstacles.push_back(Surface2D(50, 10, 50, 700));
	m_Obstacles.push_back(Surface2D(50, 699, 1200, 700));
	m_Obstacles.push_back(Surface2D(1200, 10, 1200, 700));

	//m_GeneratorPipes.push_back(GeneratorPipe(Vector2D(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 2), 5));
	//m_ConsumerPipes.push_back(ConsumerPipe(Vector2D(3 * SCREEN_WIDTH / 4, SCREEN_HEIGHT * 3 / 4), 10));

	//m_Obstacles.push_back(Surface2D(3 * SCREEN_WIDTH / 4 + 100, SCREEN_HEIGHT / 2 - 50, 3 * SCREEN_WIDTH / 4 + 100, SCREEN_HEIGHT / 2 + 50));

	/*m_Obstacles.push_back(Surface2D(500, 400, 600, 300));
	m_Obstacles.push_back(Surface2D(600, 300, 700, 400));
	m_Obstacles.push_back(Surface2D(700, 400, 500, 400));*/

	m_SolidObjects.push_back(SolidRectangle(10, 10, 0.5, Vector2D(640, 100)));

	GpuAllocate(m_Particles, m_Obstacles, interactionMatrixRows * interactionMatrixCols, m_ConsumerPipes, m_GeneratorPipes);
}

#include<windows.h>

Environment::~Environment()
{

	/*for (auto& pipe : m_Pipes) {
		delete pipe;
	}*/

	GpuFree();
}

void Environment::renderParticles(int width, int height) {
	for (auto& particle : m_Particles) {

		if (!particle.m_Exists) {
			continue;
		}

		//float density = particle->m_Density;

		Vector2D vc = particle.getVelocity();
		vc = vc / 2;

		//float color = density / maxDensity;

		float blue, green, red;

		Graphics::velocityToColor(particle.getVelocity().getMagnitude(), red, green, blue);

		glColor4f(red, green, blue, 1.0f);
		Graphics::DrawCircle(width, height, particle.getPosition().X, particle.getPosition().Y, particleRadius * 2, 20);

		/*glColor4f(1.0, 1.0, 1.0, 0.4f);
		Graphics::DrawLine(width, height, particle.m_Position, particle.m_Position + vc);*/
	}
}

void Environment::render(int width, int height)
{

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	/*float maxDensity = 0.0f;
	for (auto& particle : m_Particles) {

		if (particle.m_Density > maxDensity) {
			maxDensity = particle.m_Density;
		}
	}*/

	for (auto pipe : m_ConsumerPipes) {
		glColor4f(1.0, 1.0, 1.0, 0.5);
		Graphics::DrawCircle(width, height, pipe.getPosition().X, pipe.getPosition().Y, pipe.getInteractionRadius() * 2, 20);
	}

	for (auto pipe : m_GeneratorPipes) {
		glColor4f(1.0, 1.0, 1.0, 0.5);
		Graphics::DrawCircle(width, height, pipe.getPosition().X, pipe.getPosition().Y, pipe.getInteractionRadius() * 2, 20);
	}

	this->renderParticles(width, height);

	for (auto& obstacle : m_Obstacles) {
		glColor3f(1.0, 1.0, 1.0);
		Graphics::DrawLine(width, height, obstacle.Point1, obstacle.Point2);
	}

	// render solid objects
	for (auto& solidObject : m_SolidObjects) {
		glColor3f(1.0, 1.0, 1.0);
		Graphics::DrawRectangle(width, height, solidObject);
	}

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	int x = 0;
}

bool customCompare(Particle* a, Particle* b) {
	// Define your custom comparison logic here
	// For example, sort in descending order

	int rowA = a->m_Position.Y / particleRadiusOfRepel;
	int colA = a->m_Position.X / particleRadiusOfRepel;

	int rowB = b->m_Position.Y / particleRadiusOfRepel;
	int colB = b->m_Position.X / particleRadiusOfRepel;

	if (rowA == rowB) {
		return colA < colB;
	}

	return rowA < rowB;
}

void Environment::newUpdate(float dt) {

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	//std::sort(m_Particles.begin(), m_Particles.end(), customCompare);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	time1 = std::chrono::steady_clock::now();

	GpuUpdateParticles(m_Particles, m_ParticleCount, particleRadiusOfRepel, particleRadius, particleRepulsionForce, m_Obstacles, dt,
		interactionMatrixRows, interactionMatrixCols);

	/*for (auto& pipe : m_Pipes) {
		pipe->update(dt, m_Particles, InteractionMatrixClass::getInstance()->getParticlesInCell(pipe->getPosition(), particleRadiusOfRepel), particleRadius * 2);
	}*/

	// update solid objects
	for (auto& solidObject : m_SolidObjects) {
		solidObject.update(dt);
	}

	time2 = std::chrono::steady_clock::now();
	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	int x = 0;
}

