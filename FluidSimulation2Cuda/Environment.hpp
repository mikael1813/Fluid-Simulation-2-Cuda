#pragma once

#include "Particle.hpp"
#include "Pipe.hpp"
#include "SolidObject.hpp"

//#include "SDL.h"

#include <vector>
#include "thread"


class Environment
{
public:
	Environment(int particleCount, int particleRadius, int particleRadiusOfRepel, float particleRepulsionForce, int screenWidth,
		int screenHeight, float viscosityStrength, float how_far_into_the_future, int thread_count,
		int interactionMatrixRows, int interactionMatrixCols, std::vector<Surface2D> obstacles,
		std::vector<ConsumerPipe> consumers, std::vector<GeneratorPipe> generators, Surface2D spawnArea);
	~Environment();


	void render(int width, int height);
	void update(float dt);
	void newUpdate(float dt);

	void moveUp();
	void moveDown();
	void moveLeft();
	void moveRight();
	void turnGenerators();
	void turnSurfaceTension();
	void deleteLastObstacle();

	void updatePressureCoefficient(float pressureCoefficient);
	void updateGravity(float gravity);
	void updateTargetDensity(float targetDensity);
	void updateViscosity(float viscosity);

	std::vector<float> m_WallPressure{ 0,0,0,0 };

private:

	int m_ParticleCount = 3000;
	//int m_ParticleCount = 20000;
	//int m_ParticleCount = 10000;
	int m_ParticleRadius;
	int m_ParticleRadiusOfRepel;
	int m_ParticleRepulsionForce;
	int m_InteractionMatrixRows;
	int m_InteractionMatrixCols;

	float m_ViscosityStrength;
	float m_TargetDensity = 0.8f;
	float m_PressureCoefficient = 50.0f;

	float m_Gravity = 20.0f;

	bool m_GeneratorsTurned = false;
	bool m_ApplySurfaceTension = false;

	std::vector<Particle> m_Particles;
	std::vector<float> m_ParticleProperties;
	std::vector<float> m_ParticleDensities;

	std::vector<ConsumerPipe> m_ConsumerPipes;
	std::vector<GeneratorPipe> m_GeneratorPipes;

	std::vector<SolidRectangle> m_SolidObjects;

	std::vector<std::thread> m_Threads;

	std::vector<Vector2D> m_ExternalForces;

	float calculateDensity(Vector2D point);
	float calculateProperty(Vector2D point);

	void renderParticles(int width, int height);

	void checkCollisions(int start, int end);

	void updateParticleDensities(int start, int end);
	void calculateFutureVelocities(double dt, int start, int end);

	void parallelCheckCollisions();

	void parallelUpdateParticleDensities();
	void parallelCalculateFutureVelocities(double dt);

	Vector2D calculateViscosityForce(Particle* particle);

	Vector2D calculatePressureForce(Particle* particle);

	std::vector<Surface2D> m_Obstacles;
};