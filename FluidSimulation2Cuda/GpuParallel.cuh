#ifndef GpuParallel_h
#define GpuParallel_h

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.hpp"

#include <vector>
#include "InteractionMatrixClass.hpp"


void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, InteractionMatrixClass* interactionMatrix, int particleRadiusOfRepel);


// Define the Particle struct
struct Quo {
	// Define particle properties here
	float density = 2;
};
void processDataOnGPU(std::vector<Quo>& particles);

#endif