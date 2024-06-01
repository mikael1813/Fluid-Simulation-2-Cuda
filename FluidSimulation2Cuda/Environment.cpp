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

float ExampleFunction(Vector2D point) {
	return cos(point.Y - 3 + sin(point.X));
}


Environment::Environment(int tmp_particleCount, int particleRadius, int particleRadiusOfRepel, float particleRepulsionForce, int screenWidth,
	int screenHeight, float viscosityStrength, float how_far_into_the_future, int thread_count,
	int interactionMatrixRows, int interactionMatrixCols, std::vector<Surface2D> obstacles,
	std::vector<ConsumerPipe> consumers, std::vector<GeneratorPipe> generators) {

	m_ParticleCount = tmp_particleCount;

	m_ParticleRadius = particleRadius;
	m_ParticleRadiusOfRepel = particleRadiusOfRepel;
	m_ParticleRepulsionForce = particleRepulsionForce;
	m_InteractionMatrixRows = interactionMatrixRows;
	m_InteractionMatrixCols = interactionMatrixCols;

	InteractionMatrixClass::getInstance()->initializeMatrix(screenWidth, screenHeight, particleRadiusOfRepel);

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

			posX = std::uniform_int_distribution<int>(100, screenWidth - 100)(gen);
			posY = std::uniform_int_distribution<int>(100, screenHeight - 100)(gen);

			for (int obstacleIndex = 0; obstacleIndex < obstacles.size(); obstacleIndex += 4) {

				Vector2D topLeft = obstacles[obstacleIndex].Point1;
				Vector2D bottomRight = obstacles[obstacleIndex].Point2;

				for (int j = 0; j < 4; j++) {
					auto& obstacle = obstacles[obstacleIndex + j];

					// get the obstacles top left corner and bottom right corner
					topLeft = Vector2D(std::min(topLeft.X, obstacle.Point2.X), std::min(topLeft.Y, obstacle.Point2.Y));
					bottomRight = Vector2D(std::max(bottomRight.X, obstacle.Point2.X), std::max(bottomRight.Y, obstacle.Point2.Y));

					// check if the particle is inside the obstacle
					if (posX >= topLeft.X && posX <= bottomRight.X && posY >= topLeft.Y && posY <= bottomRight.Y) {
						ok = false;
						break;
					}
				}
			}

			if (!ok) {
				continue;
			}

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
		m_Particles.push_back(Particle(count++));
	}

	m_Obstacles.push_back(Surface2D(0, 0, screenWidth, 0));
	m_Obstacles.push_back(Surface2D(screenWidth, 0, screenWidth, screenHeight));
	m_Obstacles.push_back(Surface2D(screenWidth, screenHeight, 0, screenHeight));
	m_Obstacles.push_back(Surface2D(0, screenHeight, 0, 0));

	//m_GeneratorPipes.push_back(GeneratorPipe(Vector2D(screenWidth / 4, screenHeight / 2), 5));
	//m_ConsumerPipes.push_back(ConsumerPipe(Vector2D(3 * screenWidth / 4, screenHeight * 3 / 4), 10));

	//m_Obstacles.push_back(Surface2D(3 * screenWidth / 4 + 100, screenHeight / 2 - 50, 3 * screenWidth / 4 + 100, screenHeight / 2 + 50));

	/*m_Obstacles.push_back(Surface2D(500, 400, 600, 300));
	m_Obstacles.push_back(Surface2D(600, 300, 700, 400));
	m_Obstacles.push_back(Surface2D(700, 400, 500, 400));*/

	for (auto& obstacle : obstacles) {
		m_Obstacles.push_back(obstacle);
	}

	for (auto& consumer : consumers) {
		m_ConsumerPipes.push_back(consumer);
	}

	for (auto& generator : generators) {
		m_GeneratorPipes.push_back(generator);
	}

	//m_SolidObjects.push_back(SolidRectangle(30, 30, 0.05, Vector2D(640, 50)));

	GpuAllocate(m_Particles, m_Obstacles, interactionMatrixRows * interactionMatrixCols, m_ConsumerPipes, m_GeneratorPipes, m_SolidObjects);
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
		Graphics::DrawCircle(width, height, particle.getPosition().X, particle.getPosition().Y, m_ParticleRadius * 2, 20);

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

//bool customCompare(Particle* a, Particle* b) {
//	// Define your custom comparison logic here
//	// For example, sort in descending order
//
//	int rowA = a->m_Position.Y / m_ParticleRadiusOfRepel;
//	int colA = a->m_Position.X / m_ParticleRadiusOfRepel;
//
//	int rowB = b->m_Position.Y / m_ParticleRadiusOfRepel;
//	int colB = b->m_Position.X / m_ParticleRadiusOfRepel;
//
//	if (rowA == rowB) {
//		return colA < colB;
//	}
//
//	return rowA < rowB;
//}

void Environment::newUpdate(float dt) {

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	//std::sort(m_Particles.begin(), m_Particles.end(), customCompare);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	time1 = std::chrono::steady_clock::now();

	if (m_ExternalForces.size() > 0) {
		GpuApplyExternalForces(m_Particles, m_ExternalForces);
	}


	float averageDensity = 0.0f;
	/*for(int i = 0; i < m_ParticleCount; i++) {
		averageDensity += m_Particles[i].m_Density;
	}
	averageDensity /= m_ParticleCount;*/
	//printf("Average density: %f\n", averageDensity);

	bool resizeNeeded = false;

	GpuUpdateParticles(m_Particles, m_ParticleCount, m_ParticleRadiusOfRepel, m_ParticleRadius, m_ParticleRepulsionForce,
		m_Obstacles, m_SolidObjects, dt, m_InteractionMatrixRows, m_InteractionMatrixCols, averageDensity, m_GeneratorsTurned,
		resizeNeeded);

	if (resizeNeeded) {
		int particleCount = Math::nextPowerOf2(m_Particles.size() + 1);

		int count = m_Particles.size();

		for (int i = m_Particles.size(); i < particleCount; i++) {
			m_Particles.push_back(Particle(count++));
		}

		GpuReallocateParticles(m_Particles);
	}

	/*GpuUpdateParticles(m_Particles, m_ParticleCount, particleRadiusOfRepel, particleRadius, particleRepulsionForce,
		m_Obstacles, m_SolidObjects, dt, interactionMatrixRows, interactionMatrixCols, averageDensity);*/

		/*for (auto& pipe : m_Pipes) {
			pipe->update(dt, m_Particles, InteractionMatrixClass::getInstance()->getParticlesInCell(pipe->getPosition(), particleRadiusOfRepel), particleRadius * 2);
		}*/

		// update solid objects
		/*for (auto& solidObject : m_SolidObjects) {
			solidObject.update(dt);
		}*/

	time2 = std::chrono::steady_clock::now();
	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	int x = 0;
}

void Environment::moveUp()
{
	Vector2D force = Vector2D(0, -100);
	m_ExternalForces.push_back(force);
}

void Environment::moveDown()
{
	Vector2D force = Vector2D(0, 100);
	m_ExternalForces.push_back(force);
}

void Environment::moveLeft()
{
	Vector2D force = Vector2D(-100, 0);
	m_ExternalForces.push_back(force);
}

void Environment::moveRight()
{
	Vector2D force = Vector2D(100, 0);
	m_ExternalForces.push_back(force);
}

void Environment::turnGenerators() {
	m_GeneratorsTurned = !m_GeneratorsTurned;
}

