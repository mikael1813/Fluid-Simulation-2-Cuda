#pragma once

#include "Particle.hpp"
#include "Pipe.hpp"

//#include "SDL.h"

#include <vector>
#include "thread"


class Environment
{
public:
	Environment();
	~Environment();


	void render(int width, int height);
	void update(float dt);
	void newUpdate(float dt);

private:
	std::vector<Particle> m_Particles;
	std::vector<float> m_ParticleProperties;
	std::vector<float> m_ParticleDensities;

	std::vector<Pipe*> m_Pipes;

	std::vector<std::thread> m_Threads;

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