#ifndef GpuParallel_h
#define GpuParallel_h

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.hpp"

#include <vector>
#include "InteractionMatrixClass.hpp"

#include <curand_kernel.h>

void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, int particleRadiusOfRepel);

void GpuParallelCalculateFutureVelocities(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, double dt);

void GpuParallelCheckCollision(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, float repulsionForce, std::vector<Surface2D>& obstacles);

void GpuUpdateParticles(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles,
	double dt, size_t interactionMatrixRows, size_t interactionMatrixCols,
	InteractionMatrixClass* interactionMatrix);

void GpuAllocateInteractionMatrix(InteractionMatrixClass* interactionMatrix);

void GpuFreeInteractionMatrix();

void GpuAllocate(std::vector<Particle>& particles, std::vector<Surface2D>& obstacles, int interactionMatrixSize);

void GpuFree();

#endif