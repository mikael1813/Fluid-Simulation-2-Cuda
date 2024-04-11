#include "GpuParallel.cuh"

#include "Phisics.hpp"

__global__ void updateParticleDensitiesKernel(Particle* particles, int particleRadiusOfRepel) {

	int index = threadIdx.x;

	printf("index: %d", index);

	/*Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	particles[index].m_Density = 2.8;*/

	/*std::vector<Particle*> particlesInCell = interactionMatrix->getParticlesInCell(point, particleRadiusOfRepel);

	constexpr auto scalar = 1000;

	float density = 0.0f;
	const float mass = 1.0f;

	for (int i = 0; i < particlesInCell.size(); i++) {
		float distance = sqrt(Math::squared_distance(point, particle->m_PredictedPosition));
		float influence = Math::smoothingKernel(particleRadiusOfRepel, distance);
		density += mass * influence;
	}

	float volume = 3.1415f * pow(particleRadiusOfRepel, 2);

	density = density / volume * scalar;

	particle->m_Density = density;*/
}



void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, InteractionMatrixClass* interactionMatrix, int particleRadiusOfRepel) {


	// Allocate memory on GPU
	Particle* gpuParticles;
	cudaMalloc(&gpuParticles, particles.size() * sizeof(Particle));

	// Copy data from CPU to GPU
	cudaMemcpy(gpuParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	// Launch CUDA kernel
	updateParticleDensitiesKernel << <1, particles.size() >> > (gpuParticles, particles.size());

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, gpuParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free GPU memory
	cudaFree(gpuParticles);

	//Particle* cudaParticles;

	//cudaError_t cudaStatus;

	//InteractionMatrixClass* cudaInteractionMatrix;
	////int cudaParticleRadiusOfRepel;

	//cudaStatus = cudaMalloc(&cudaParticles, particles.size() * sizeof(Particle));

	//cudaStatus = cudaMemcpy(cudaParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	//updateParticleDensitiesKernel << <1, particles.size() >> > (cudaParticles, particleRadiusOfRepel);

	//// Wait for the kernel to finish
	//cudaDeviceSynchronize();

	//Particle* resultParticles = new Particle[particles.size()];
	//cudaMemcpy(resultParticles, cudaParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	////cudaStatus = cudaMemcpy(output, cudaParticles, particles.size() * sizeof(Particle*), cudaMemcpyDeviceToHost);
	//auto y = resultParticles[9];
	//int x = 0;

	//// Cleanup resources
	//cudaFree(cudaParticles);

	//x = 2;

}

// CUDA kernel function
__global__ void processParticlesKernel(Quo* particles, int numParticles) {
	int index = threadIdx.x;
	printf("index: %d\n", index);
	if (index < numParticles) {
		particles[index].density = index; // Process particle data here
	}
}

void processDataOnGPU(std::vector<Quo>& particles) {
	// Allocate memory on GPU
	Quo* gpuParticles;
	cudaMalloc(&gpuParticles, particles.size() * sizeof(Quo));

	// Copy data from CPU to GPU
	cudaMemcpy(gpuParticles, particles.data(), particles.size() * sizeof(Quo), cudaMemcpyHostToDevice);

	// Launch CUDA kernel
	processParticlesKernel << <1, particles.size() >> > (gpuParticles, particles.size());

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	Quo* output = new Quo[particles.size()];

	cudaMemcpy(output, gpuParticles, particles.size() * sizeof(Quo), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free GPU memory
	cudaFree(gpuParticles);
}