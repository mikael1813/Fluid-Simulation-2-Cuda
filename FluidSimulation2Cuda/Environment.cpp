#include "Environment.hpp"
#include "InteractionMatrixClass.hpp"
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


constexpr auto particleCount = 4096;

constexpr auto particleRadius = 2;
constexpr auto particleRadiusOfRepel = 50;
constexpr auto particleDistance = 30;

constexpr auto particleRepulsionForce = 1.0f;

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

	for (int i = 0; i < particleCount; i++) {

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

	m_Obstacles.push_back(Surface2D(50, 10, 1200, 11));
	m_Obstacles.push_back(Surface2D(50, 10, 50, 700));
	m_Obstacles.push_back(Surface2D(50, 699, 1200, 700));
	m_Obstacles.push_back(Surface2D(1200, 10, 1200, 700));

	/*m_Pipes.push_back(new GeneratorPipe(Vector2D(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 2), 5));
	m_Pipes.push_back(new ConsumerPipe(Vector2D(3 * SCREEN_WIDTH / 4, SCREEN_HEIGHT / 2), 10));*/

	//m_Obstacles.push_back(Surface2D(3 * SCREEN_WIDTH / 4 + 100, SCREEN_HEIGHT / 2 - 50, 3 * SCREEN_WIDTH / 4 + 100, SCREEN_HEIGHT / 2 + 50));

	/*m_Obstacles.push_back(Surface2D(500, 400, 600, 300));
	m_Obstacles.push_back(Surface2D(600, 300, 700, 400));
	m_Obstacles.push_back(Surface2D(700, 400, 500, 400));*/

	/*std::vector<Particle> temporaryParticles;

	for (auto& particle : m_Particles) {
		temporaryParticles.push_back(particle);
	}*/

	GpuAllocate(m_Particles, m_Obstacles, interactionMatrixRows * interactionMatrixCols);
}

#include<windows.h>

Environment::~Environment()
{

	for (auto& pipe : m_Pipes) {
		delete pipe;
	}

	GpuFree();
}

void Environment::renderParticles(int width, int height) {
	for (auto& particle : m_Particles) {

		//float density = particle->m_Density;

		Vector2D vc = particle.getVelocity();

		//float color = density / maxDensity;

		float blue, green, red;

		Graphics::velocityToColor(particle.getVelocity().getMagnitude(), red, green, blue);

		glColor4f(red, green, blue, 1.0f);
		Graphics::DrawCircle(width, height, particle.getPosition().X, particle.getPosition().Y, particleRadius * 2, 20);

		/*glColor4f(1.0, 1.0, 1.0, 0.4f);
		DrawLine(width, height, particle.m_Position, particle.m_Position + vc);*/
	}
}

void Environment::render(int width, int height)
{
	/*float maxDensity = 0.0f;
	for (auto& particle : m_Particles) {

		if (particle.m_Density > maxDensity) {
			maxDensity = particle.m_Density;
		}
	}*/

	for (auto& pipe : m_Pipes) {
		glColor4f(1.0, 1.0, 1.0, 0.5);
		Graphics::DrawCircle(width, height, pipe->getPosition().X, pipe->getPosition().Y, pipe->getInteractionRadius() * 2, 20);
	}

	this->renderParticles(width, height);

	for (auto& obstacle : m_Obstacles) {
		glColor3f(1.0, 1.0, 1.0);
		Graphics::DrawLine(width, height, obstacle.Point1, obstacle.Point2);
	}
}

//void Environment::update(float dt) {
//
//	//std::cout<<"A"<<std::endl;
//
//	//std::cout<<"B"<<std::endl;
//
//	// calculate predicted positions
//	for (auto& particle : m_Particles) {
//		particle->m_PredictedPosition = particle->getPosition() + particle->getVelocity() * dt * HOW_FAR_INTO_THE_FUTURE;
//	}
//
//	//std::cout<<"C"<<std::endl;
//
//	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
//	InteractionMatrixClass::getInstance()->updateInteractionMatrix(m_Particles, particleRadiusOfRepel);
//
//
//	//std::cout<<"D"<<std::endl;
//
//	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
//	double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
//
//	for (auto& pipe : m_Pipes) {
//		pipe->update(dt, m_Particles, InteractionMatrixClass::getInstance()->getParticlesInCell(pipe->getPosition(), particleRadiusOfRepel), particleRadius * 2);
//	}
//
//	std::vector<Particle> temporaryParticles;
//
//	time1 = std::chrono::steady_clock::now();
//
//	//this->updateParticleDensities(0, m_Particles.size());
//	//this->parallelUpdateParticleDensities();
//
//	GpuAllocateInteractionMatrix(InteractionMatrixClass::getInstance());
//
//	for (auto& particle : m_Particles) {
//		temporaryParticles.push_back(*particle);
//	}
//
//	GpuParallelUpdateParticleDensities(temporaryParticles, particleRadiusOfRepel);
//
//	GpuFreeInteractionMatrix();
//
//	for (int i = 0; i < m_Particles.size(); i++) {
//		m_Particles.at(i)->m_Density = temporaryParticles.at(i).m_Density;
//	}
//
//	time2 = std::chrono::steady_clock::now();
//	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
//
//	time1 = std::chrono::steady_clock::now();
//
//	//this->calculateFutureVelocities(dt, 0, m_Particles.size());
//	//this->parallelCalculateFutureVelocities(dt);
//
//	temporaryParticles.clear();
//
//	for (auto& particle : m_Particles) {
//		temporaryParticles.push_back(*particle);
//	}
//
//	GpuAllocateInteractionMatrix(InteractionMatrixClass::getInstance());
//
//	GpuParallelCalculateFutureVelocities(temporaryParticles, particleRadiusOfRepel, particleRadius, dt);
//
//	GpuFreeInteractionMatrix();
//
//	for (int i = 0; i < m_Particles.size(); i++) {
//		m_Particles.at(i)->m_FutureVelocity = temporaryParticles.at(i).m_FutureVelocity;
//		m_Particles.at(i)->m_Velocity = temporaryParticles.at(i).m_Velocity;
//		m_Particles.at(i)->m_Position = temporaryParticles.at(i).m_Position;
//		m_Particles.at(i)->m_TemporaryVelocity = temporaryParticles.at(i).m_TemporaryVelocity;
//		m_Particles.at(i)->m_PredictedPosition = temporaryParticles.at(i).m_PredictedPosition;
//		m_Particles.at(i)->m_LastSafePosition = temporaryParticles.at(i).m_LastSafePosition;
//	}
//
//	time2 = std::chrono::steady_clock::now();
//	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
//
//	//time1 = std::chrono::steady_clock::now();
//
//	//std::cout<<"F"<<std::endl;
//
//	// apply future velocities to current velocities
//	/*for (auto& particle : m_Particles) {
//		particle->updateVelocity();
//		particle->update(dt);
//		particle->m_PredictedPosition = particle->getPosition();
//	}*/
//
//	//std::cout<<"G"<<std::endl;
//
//	//InteractionMatrixClass::getInstance()->updateInteractionMatrix(m_Particles, particleRadiusOfRepel);
//
//	//std::cout<<"H"<<std::endl;
//
//	time1 = std::chrono::steady_clock::now();
//
//	//this->checkCollisions(0, m_Particles.size());
//	//this->parallelCheckCollisions();
//
//
//	temporaryParticles.clear();
//
//	for (auto& particle : m_Particles) {
//		temporaryParticles.push_back(*particle);
//	}
//
//
//	GpuAllocateInteractionMatrix(InteractionMatrixClass::getInstance());
//
//	//GpuParallelCheckCollision(temporaryParticles, particleRadiusOfRepel, particleRadius, particleRepulsionForce, m_Obstacles);
//
//	GpuFreeInteractionMatrix();
//
//	for (int i = 0; i < m_Particles.size(); i++) {
//		m_Particles.at(i)->m_Velocity = temporaryParticles.at(i).m_Velocity;
//		m_Particles.at(i)->m_TemporaryVelocity = temporaryParticles.at(i).m_TemporaryVelocity;
//		m_Particles.at(i)->m_Position = temporaryParticles.at(i).m_Position;
//	}
//
//	time2 = std::chrono::steady_clock::now();
//	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
//
//	int x = 0;
//}

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

	/*std::vector<Particle> temporaryParticles;

	for (auto& particle : m_Particles) {
		temporaryParticles.push_back(*particle);
	}*/

	/*int interactionMatrixRows = SCREEN_HEIGHT / particleRadiusOfRepel;
	int interactionMatrixCols = SCREEN_WIDTH / particleRadiusOfRepel;*/

	time1 = std::chrono::steady_clock::now();

	GpuUpdateParticles(m_Particles, particleRadiusOfRepel, particleRadius, particleRepulsionForce, m_Obstacles, dt,
		interactionMatrixRows, interactionMatrixCols, InteractionMatrixClass::getInstance());

	/*for (int i = 0; i < m_Particles.size(); i++) {
		m_Particles.at(i)->m_Density = temporaryParticles.at(i).m_Density;
		m_Particles.at(i)->m_FutureVelocity = temporaryParticles.at(i).m_FutureVelocity;
		m_Particles.at(i)->m_Velocity = temporaryParticles.at(i).m_Velocity;
		m_Particles.at(i)->m_Position = temporaryParticles.at(i).m_Position;
		m_Particles.at(i)->m_TemporaryVelocity = temporaryParticles.at(i).m_TemporaryVelocity;
		m_Particles.at(i)->m_PredictedPosition = temporaryParticles.at(i).m_PredictedPosition;
		m_Particles.at(i)->m_LastSafePosition = temporaryParticles.at(i).m_LastSafePosition;
	}*/

	/*for (auto& pipe : m_Pipes) {
		pipe->update(dt, m_Particles, InteractionMatrixClass::getInstance()->getParticlesInCell(pipe->getPosition(), particleRadiusOfRepel), particleRadius * 2);
	}*/

	time2 = std::chrono::steady_clock::now();
	tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	int x = 0;
}

