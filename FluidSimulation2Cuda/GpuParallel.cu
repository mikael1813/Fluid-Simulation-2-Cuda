#include "GpuParallel.cuh"
#include "CudaMath.cuh"

#include <chrono>
#include "Phisics.hpp"

Particle* deviceInteractionMatrixParticles;

int* deviceLengths;
int interactionMatrixSize;

size_t maxParticlesInInteractionMatrixCell;
size_t interactionMatrixRows;
size_t interactionMatrixCols;

struct Range {
	int start;
	int end;
};

__device__ Range getParticlesInCell(Vector2D position, int particleRadiusOfRepel,
	int* lengths, size_t interactionMatrixRows, size_t interactionMatrixCols,
	size_t maxParticlesInInteractionMatrixCell)
{
	int row = position.Y / particleRadiusOfRepel;
	int col = position.X / particleRadiusOfRepel;

	if (row < 0 || row >= interactionMatrixRows || col < 0 || col >= interactionMatrixCols) {
		return;
	}

	int start = (row * interactionMatrixCols + col) * maxParticlesInInteractionMatrixCell;
	int end = start + lengths[row * interactionMatrixCols + col];

	return Range{ start, end };
}

__global__ void updateParticleDensitiesKernel(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	Particle* interactionMatrixParticles, int* lengths, size_t interactionMatrixRows, size_t interactionMatrixCols,
	size_t maxParticlesInInteractionMatrixCell) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", index);

	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	constexpr auto scalar = 1000;

	float density = 0.0f;
	const float mass = 1.0f;

	Range range = getParticlesInCell(point, particleRadiusOfRepel, lengths, interactionMatrixRows, interactionMatrixCols, maxParticlesInInteractionMatrixCell);

	//printf("index: %d, range.start: %d, range.end: %d \n", index, range.start, range.end);

	for (int i = range.start; i < range.end; i++) {
		Particle otherParticle = interactionMatrixParticles[i];
		float distance = sqrt(CudaMath::squared_distance(point, otherParticle.m_PredictedPosition));
		float influence = CudaMath::smoothingKernel(particleRadiusOfRepel, distance);
		density += mass * influence;
	}

	float volume = 3.1415f * pow(particleRadiusOfRepel, 2);

	density = density / volume * scalar;

	particles[index].m_Density = density;
}

__device__ GpuVector2D calculatePressureForce(Particle particle, int particleRadiusOfRepel, int particleRadius,
	Particle* interactionMatrixParticles, int* lengths, int interactionMatrixRows, int interactionMatrixCols,
	int maxParticlesInInteractionMatrixCell)
{

	GpuVector2D pressureForce = GpuVector2D();
	const float mass = 1.0f;

	Range range = getParticlesInCell(particle.m_PredictedPosition, particleRadiusOfRepel, lengths, interactionMatrixRows, interactionMatrixCols, maxParticlesInInteractionMatrixCell);

	for (int i = range.start; i < range.end; i++) {
		Particle otherParticle = interactionMatrixParticles[i];

		if (particle.m_ID == otherParticle.m_ID) {
			continue;
		}

		float distance = sqrt(CudaMath::squared_distance(particle.m_PredictedPosition, otherParticle.m_PredictedPosition));

		GpuVector2D dir = distance < particleRadius ? GpuVector2D::getRandomDirection() : (GpuVector2D(otherParticle.m_PredictedPosition) - GpuVector2D(particle.m_PredictedPosition)) / distance;

		float slope = CudaMath::smoothingKernelDerivative(particleRadiusOfRepel, distance);

		float density = otherParticle.m_Density;

		float sharedPressure = CudaMath::calculateSharedPressure(density, otherParticle.m_Density);

		pressureForce += -sharedPressure * dir * slope * mass / density;
	}
	//printf("pressureForceX: %f , pressureForceY: %f\n", pressureForce.X, pressureForce.Y);
	return pressureForce;
}


__global__ void calculateParticleFutureVelocitiesKernel(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, Particle* interactionMatrixParticles, int* lengths, size_t interactionMatrixRows,
	size_t interactionMatrixCols, size_t maxParticlesInInteractionMatrixCell, double dt)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", interactionMatrixCols);

	Particle particle = particles[index];

	if (particle.m_Density == 0) {
		return;
	}

	GpuVector2D pressureForce = calculatePressureForce(particle, particleRadiusOfRepel, particleRadius,
		interactionMatrixParticles, lengths, interactionMatrixRows, interactionMatrixCols,
		maxParticlesInInteractionMatrixCell);

	//pressureForce = GpuVector2D(300, 300);
	GpuVector2D pressureAcceleration = pressureForce / particle.m_Density;

	//Vector2D viscosityForce = calculateViscosityForce(particle);
	GpuVector2D viscosityForce = GpuVector2D();

	GpuVector2D futureVelocity = GpuVector2D(particle.m_Velocity) + pressureAcceleration * dt + viscosityForce * dt;

	//printf("index: %d, futureVelocity: %f %f \n", index, futureVelocity.X, futureVelocity.Y);

	particles[index].m_FutureVelocity.X = futureVelocity.X;
	particles[index].m_FutureVelocity.Y = futureVelocity.Y;
}

__global__ void checkCollisionsKernel(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Particle* interactionMatrixParticles, int* lengths,
	int interactionMatrixRows, int interactionMatrixCols, int maxParticlesInInteractionMatrixCell,
	Surface2D* obstacles, int obstaclesSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", interactionMatrixCols);

	Particle particle = particles[index];

	for (int i = 0; i < obstaclesSize; i++) {

		Surface2D obstacle = obstacles[i];

		if (CudaMath::check_line_segment_circle_intersection(obstacle.Point1, obstacle.Point2,
			particle.m_Position, particleRadius)) {

			//particle->m_Velocity = reflectionVector * 0.1f;
			particles[index].m_Velocity.X = 0;
			particles[index].m_Velocity.Y = 0;

			/*particle.m_Velocity.Y = -particle.m_Velocity.Y;*/

			particles[index].m_Position = particle.m_LastSafePosition;

			break;
		}
	}

	Range range = getParticlesInCell(particle.m_PredictedPosition, particleRadiusOfRepel, lengths, interactionMatrixRows, interactionMatrixCols, maxParticlesInInteractionMatrixCell);

	//for (auto& otherParticle : m_Particles) {
	for (int i = range.start; i < range.end; i++) {

		Particle otherParticle = interactionMatrixParticles[i];

		if (particle.m_ID == otherParticle.m_ID) {
			continue;
		}

		if (CudaMath::squared_distance(particle.m_Position, otherParticle.m_Position) <= (particleRadius * particleRadius) * 4) {

			GpuVector2D normalVector{};
			normalVector.X = otherParticle.m_Position.X - particle.m_Position.X;
			normalVector.Y = otherParticle.m_Position.Y - particle.m_Position.Y;

			//magnitude of normal vector
			float magnitude = -1 * sqrt(normalVector.X * normalVector.X + normalVector.Y * normalVector.Y);

			GpuVector2D temporaryVelocity = -normalVector * particleRepulsionForce;

			particles[index].m_TemporaryVelocity.X = temporaryVelocity.X;
			particles[index].m_TemporaryVelocity.Y = temporaryVelocity.Y;

			//otherParticle->m_TemporaryVelocity = normalVector * particleRepulsionForce;
		}
	}
}

void GpuAllocateInteractionMatrix(InteractionMatrixClass* interactionMatrix) {
	//

	interactionMatrixSize = interactionMatrix->getMatrix().size() * interactionMatrix->getMatrix().at(0).size();

	maxParticlesInInteractionMatrixCell = 0;
	interactionMatrixRows = interactionMatrix->getMatrix().size();
	interactionMatrixCols = interactionMatrix->getMatrix().at(0).size();

	int* hostLengths = new int[interactionMatrixSize];

	for (int i = 0; i < interactionMatrixRows; i++) {
		for (int j = 0; j < interactionMatrixCols; j++) {
			hostLengths[i * interactionMatrixCols + j] = interactionMatrix->getMatrix().at(i).at(j).particles.size();
			if (interactionMatrix->getMatrix().at(i).at(j).particles.size() > maxParticlesInInteractionMatrixCell) {
				maxParticlesInInteractionMatrixCell = interactionMatrix->getMatrix().at(i).at(j).particles.size();
			}
		}
	}

	Particle* hostInteractionMatrixParticles = new Particle[interactionMatrixSize * maxParticlesInInteractionMatrixCell];

	for (int i = 0; i < interactionMatrixRows; i++) {
		for (int j = 0; j < interactionMatrixCols; j++) {
			int index = i * interactionMatrixCols + j;

			for (int k = 0; k < maxParticlesInInteractionMatrixCell; k++) {
				if (k < hostLengths[index]) {
					hostInteractionMatrixParticles[index * maxParticlesInInteractionMatrixCell + k] =
						*interactionMatrix->getMatrix().at(i).at(j).particles[k];
				}
				else {
					hostInteractionMatrixParticles[index * maxParticlesInInteractionMatrixCell + k] = Particle();
				}
			}
		}
	}

	// Allocate memory on GPU
	cudaMalloc(&deviceLengths, interactionMatrixSize * sizeof(int));
	cudaMemcpy(deviceLengths, hostLengths, interactionMatrixSize * sizeof(int), cudaMemcpyHostToDevice);



	cudaMalloc(&deviceInteractionMatrixParticles,
		interactionMatrixSize * maxParticlesInInteractionMatrixCell * sizeof(Particle));

	// Copy data from CPU to GPU
	cudaMemcpy(deviceInteractionMatrixParticles, hostInteractionMatrixParticles,
		interactionMatrixSize * maxParticlesInInteractionMatrixCell * sizeof(Particle), cudaMemcpyHostToDevice);

	// Free pointers
	delete[] hostLengths;
	delete[] hostInteractionMatrixParticles;
}

void GpuFreeInteractionMatrix() {
	cudaFree(deviceInteractionMatrixParticles);
	cudaFree(deviceLengths);
}

void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, int particleRadiusOfRepel) {

	// Allocate memory on GPU
	Particle* gpuParticles;

	cudaMalloc(&gpuParticles, particles.size() * sizeof(Particle));

	// Copy data from CPU to GPU
	cudaMemcpy(gpuParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	int numThreads = particles.size();
	int maxThreadsPerBlock = 1024;

	int blockSize = maxThreadsPerBlock;
	int numBlocks = (numThreads + blockSize - 1) / blockSize;

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	// Launch CUDA kernel
	updateParticleDensitiesKernel << <numBlocks, blockSize >> > (gpuParticles, particles.size(), particleRadiusOfRepel,
		deviceInteractionMatrixParticles, deviceLengths, interactionMatrixRows, interactionMatrixCols,
		maxParticlesInInteractionMatrixCell);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	double tick = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

	// Using std::unique_ptr to manage memory
	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, gpuParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free output
	delete[] output;

	// Free GPU memory
	cudaFree(gpuParticles);

}


void GpuParallelCalculateFutureVelocities(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, double dt)
{

	// Allocate memory on GPU
	Particle* gpuParticles;

	cudaMalloc(&gpuParticles, particles.size() * sizeof(Particle));

	// Copy data from CPU to GPU
	cudaMemcpy(gpuParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	int numThreads = particles.size();
	int maxThreadsPerBlock = 1024;

	int blockSize = maxThreadsPerBlock;
	int numBlocks = (numThreads + blockSize - 1) / blockSize;

	// Launch CUDA kernel
	calculateParticleFutureVelocitiesKernel << <numBlocks, blockSize >> > (gpuParticles, particles.size(), particleRadiusOfRepel,
		particleRadius, deviceInteractionMatrixParticles, deviceLengths, interactionMatrixRows, interactionMatrixCols,
		maxParticlesInInteractionMatrixCell, dt);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Using std::unique_ptr to manage memory
	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, gpuParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free output
	delete[] output;

	// Free GPU memory
	cudaFree(gpuParticles);

}

void GpuParallelCheckCollision(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles)
{

	// Allocate memory on GPU
	Particle* deviceParticles;
	Surface2D* deviceObstacles;

	cudaMalloc(&deviceParticles, particles.size() * sizeof(Particle));
	cudaMalloc(&deviceObstacles, obstacles.size() * sizeof(Surface2D));

	// Copy data from CPU to GPU
	cudaMemcpy(deviceParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceObstacles, obstacles.data(), obstacles.size() * sizeof(Surface2D), cudaMemcpyHostToDevice);

	int numThreads = particles.size();
	int maxThreadsPerBlock = 1024;

	int blockSize = maxThreadsPerBlock;
	int numBlocks = (numThreads + blockSize - 1) / blockSize;

	// Launch CUDA kernel
	checkCollisionsKernel << <numBlocks, blockSize >> > (deviceParticles, particles.size(), particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, deviceInteractionMatrixParticles, deviceLengths, interactionMatrixRows,
		interactionMatrixCols, maxParticlesInInteractionMatrixCell, deviceObstacles, obstacles.size());

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Using std::unique_ptr to manage memory
	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, deviceParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free output
	delete[] output;

	// Free GPU memory
	cudaFree(deviceParticles);
	cudaFree(deviceObstacles);

}

