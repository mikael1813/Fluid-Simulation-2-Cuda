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

void GpuAllocateInteractionMatrix(InteractionMatrixClass* interactionMatrix);

void GpuFreeInteractionMatrix();

#endif