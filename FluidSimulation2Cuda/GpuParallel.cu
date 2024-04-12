#include "GpuParallel.cuh"
#include "CudaMath.cuh"

#include "Phisics.hpp"

struct GpuMatrixComponents {
	Particle* particles;
	unsigned long long int size;
};

struct GpuInteractionMatrix {
	GpuMatrixComponents** matrix;
	unsigned long long int width;
	unsigned long long int height;


	__device__ GpuMatrixComponents getParticlesInCell(Vector2D particlePosition, int particleRadiusOfRepel) {
		int x = particlePosition.Y / particleRadiusOfRepel;
		int y = particlePosition.X / particleRadiusOfRepel;

		if (x < 0 || x >= width || y < 0 || y >= height) {
			return GpuMatrixComponents{};
		}

		return matrix[x][y];
	}
};

__global__ void updateParticleDensitiesKernel(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	Particle* interactionMatrix, size_t pitch, int* widths) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", index);
	printf("pitch: %d \n", pitch);

	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	constexpr auto scalar = 1000;

	float density = 0.0f;
	const float mass = 1.0f;

	for (int i = 0; i < praticlesSize; i++) {
		Particle otherParticle = particles[i];
		float distance = sqrt(CudaMath::squared_distance(point, otherParticle.m_PredictedPosition));
		float influence = CudaMath::smoothingKernel(particleRadiusOfRepel, distance);
		density += mass * influence;
	}

	float volume = 3.1415f * pow(particleRadiusOfRepel, 2);

	density = density / volume * scalar;

	particles[index].m_Density = density;
}



void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, InteractionMatrixClass* interactionMatrix, int particleRadiusOfRepel) {


	// Allocate memory on GPU
	Particle* gpuParticles;

	cudaMalloc(&gpuParticles, particles.size() * sizeof(Particle));

	// Copy data from CPU to GPU
	cudaMemcpy(gpuParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);


	//
	int interactionMatrixSize = interactionMatrix->getMatrix().size() * interactionMatrix->getMatrix().at(0).size();

	int* lengths = new int[interactionMatrixSize];
	size_t max_width = 0;
	size_t height = interactionMatrixSize;

	for (int i = 0; i < interactionMatrix->getMatrix().size(); i++) {
		for (int j = 0; j < interactionMatrix->getMatrix().at(0).size(); j++) {
			lengths[i * interactionMatrix->getMatrix().at(0).size() + j] = interactionMatrix->getMatrix().at(i).at(j).particles.size();
			if (interactionMatrix->getMatrix().at(i).at(j).particles.size() > max_width) {
				max_width = interactionMatrix->getMatrix().at(i).at(j).particles.size();
			}
		}
	}

	Particle** particlesInMatrix = new Particle * [height];

	for (int i = 0; i < interactionMatrix->getMatrix().size(); i++) {
		for (int j = 0; j < interactionMatrix->getMatrix().at(0).size(); j++) {
			int index = i * interactionMatrix->getMatrix().at(0).size() + j;
			printf("index: %d \n", index);
			particlesInMatrix[index] = new Particle[max_width];
			int xx = lengths[index];
			int yy = lengths[16];
			for (int k = 0; k < max_width; k++) {
				if (k < lengths[index]) {
					particlesInMatrix[index][k] = *interactionMatrix->getMatrix().at(i).at(j).particles[k];
				}
				else {
					particlesInMatrix[index][k] = Particle();
				}
			}
		}
	}

	// q
	int Nrows = 10;
	int Ncols = 10;
	//float hostPtr[Nrows][Ncols];
	float** hostPtr = new float*[Nrows];
	for (int i = 0; i < Nrows; i++) {
		hostPtr[i] = new float[Ncols];
	}
	float* devPtr;
	size_t pitch;

	for (int i = 0; i < Nrows; i++)
		for (int j = 0; j < Ncols; j++) {
			hostPtr[i][j] = 1.f;
			//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
		}

	// --- 2D pitched allocation and host->device memcopy
	cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows);
	cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols * sizeof(float), Ncols * sizeof(float), Nrows, cudaMemcpyHostToDevice);

	//q

	Particle* gpuInteractionMatrixParticles;
	size_t gpuPitch;
	int* gpuLengths;

	cudaMallocPitch(&gpuInteractionMatrixParticles, &gpuPitch, max_width * sizeof(Particle), height);
	cudaMalloc(&gpuLengths, height * sizeof(int));

	// Copy data and widths from host to device
	cudaMemcpy2D(gpuInteractionMatrixParticles, gpuPitch, particlesInMatrix, max_width * sizeof(Particle), max_width * sizeof(Particle), height, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuLengths, lengths, height * sizeof(int), cudaMemcpyHostToDevice);

	//

	int numThreads = particles.size();
	int maxThreadsPerBlock = 1024;

	int blockSize = maxThreadsPerBlock;
	int numBlocks = (numThreads + blockSize - 1) / blockSize;

	// Launch CUDA kernel
	updateParticleDensitiesKernel << <numBlocks, blockSize >> > (gpuParticles, particles.size(), particleRadiusOfRepel, gpuInteractionMatrixParticles,
		gpuPitch, gpuLengths);

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