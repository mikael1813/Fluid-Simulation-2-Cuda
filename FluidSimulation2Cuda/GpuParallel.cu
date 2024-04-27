#include "GpuParallel.cuh"
#include "CudaMath.cuh"

#include <chrono>
#include "Phisics.hpp"

#include <cuda_runtime.h>

constexpr float HOW_FAR_INTO_THE_FUTURE = 10.0f;

constexpr int maxThreadsPerBlock = 512;

struct Range {
	int start;
	int end;
};


Particle* deviceParticles;

Range* lengths;

Surface2D* deviceObstacles;

Particle* deviceInteractionMatrixParticles;

int* deviceLengths;
int interactionMatrixSize;

size_t maxParticlesInInteractionMatrixCell;
size_t interactionMatrixRows;
size_t interactionMatrixCols;

__device__ int counterDensitiesDone = 0;
__device__ int counterPredictedPositionsDone = 0;
__device__ int counterFutureVelocitiesDone = 0;


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

__global__ void updateParticleDensitiesKernel2(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	Particle* interactionMatrixParticles, int* lengths, size_t interactionMatrixRows, size_t interactionMatrixCols,
	size_t maxParticlesInInteractionMatrixCell) {

	//int index = blockIdx.x * blockDim.x + threadIdx.x;

	int index = blockIdx.x;

	int otherParticleIndex = threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", index);

	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	constexpr auto scalar = 1000;

	//float density = 0.0f;
	const float mass = 1.0f;

	Range range = getParticlesInCell(point, particleRadiusOfRepel, lengths, interactionMatrixRows, interactionMatrixCols, maxParticlesInInteractionMatrixCell);

	//printf("index: %d, range.start: %d, range.end: %d \n", index, range.start, range.end);

	int i = range.start + otherParticleIndex;

	if (i >= range.end) {
		return;
	}

	__shared__ float sharedDensity[1024];

	Particle otherParticle = interactionMatrixParticles[i];
	float distance = sqrt(CudaMath::squared_distance(point, otherParticle.m_PredictedPosition));
	float influence = CudaMath::smoothingKernel(particleRadiusOfRepel, distance);

	float localDensity = mass * influence;

	//particles[index].m_Density += localDensity;
	atomicAdd(&particles[index].m_Density, localDensity);

	// Synchronize all threads in the block
	__syncthreads();

	if (otherParticleIndex != 0) {
		return;
	}

	float volume = 3.1415f * pow(particleRadiusOfRepel, 2);

	particles[index].m_Density = particles[index].m_Density / volume * scalar;

	//printf("index: %d, density: %f \n", index, particles[index].m_Density);
}

__device__ void updateParticle(int index, Particle* particle, double dt) {
	if (dt == 0) {
		return;
	}

	particle[index].m_LastSafePosition = particle[index].m_Position;

	GpuVector2D newVelocity{ 0,0 };

	newVelocity = GpuVector2D(particle[index].m_Velocity) + GpuVector2D(0.0f, GRAVITY) * dt;

	newVelocity += GpuVector2D(particle[index].m_TemporaryVelocity);

	particle[index].m_Velocity.X = newVelocity.X;
	particle[index].m_Velocity.Y = newVelocity.Y;

	particle[index].m_TemporaryVelocity.X = 0;
	particle[index].m_TemporaryVelocity.Y = 0;

	particle[index].m_Position.X += particle[index].m_Velocity.X * dt;
	particle[index].m_Position.Y += particle[index].m_Velocity.Y * dt;
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


	particles[index].m_Velocity = particles[index].m_FutureVelocity;
	updateParticle(index, particles, dt);
	particles[index].m_PredictedPosition = particles[index].m_Position;
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

__device__ void updateParticleDensities(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	Range* lengths, int interactionMatrixRows, int interactionMatrixCols)
{

	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	constexpr auto scalar = 1000;

	float density = 0.0f;
	const float mass = 1.0f;

	int row = point.Y / particleRadiusOfRepel;
	int col = point.X / particleRadiusOfRepel;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if (row + i < 0 || row + i >= interactionMatrixRows || col + j < 0 || col + j >= interactionMatrixCols) {
				continue;
			}

			int lengthIndex = (row + i) * interactionMatrixCols + col + j;

			for (int otherParticleIndex = lengths[lengthIndex].start; otherParticleIndex < lengths[lengthIndex].end; otherParticleIndex++) {
				Particle otherParticle = particles[otherParticleIndex];
				float distance = sqrt(CudaMath::squared_distance(point, otherParticle.m_PredictedPosition));
				float influence = CudaMath::smoothingKernel(particleRadiusOfRepel, distance);
				density += mass * influence;
			}

		}
	}

	float volume = 3.1415f * pow(particleRadiusOfRepel, 2);

	if (density == 0) {
		//printf("densityyyyyyy1: %f, index: %d\n", density, index);
		density = mass * CudaMath::smoothingKernel(particleRadiusOfRepel, 0);
	}

	density = density / volume * scalar;

	/*if (density == 0) {
		printf("densityyyyyyy: %f, index: %d\n", density, index);
	}*/

	particles[index].m_Density = density;
}

__device__ GpuVector2D calculatePressureForce2(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, Range* lengths, int interactionMatrixRows, int interactionMatrixCols)
{

	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	int row = point.Y / particleRadiusOfRepel;
	int col = point.X / particleRadiusOfRepel;


	GpuVector2D pressureForce = GpuVector2D();
	const float mass = 1.0f;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if (row + i < 0 || row + i >= interactionMatrixRows || col + j < 0 || col + j >= interactionMatrixCols) {
				continue;
			}

			int lengthIndex = (row + i) * interactionMatrixCols + col + j;

			for (int otherParticleIndex = lengths[lengthIndex].start; otherParticleIndex < lengths[lengthIndex].end; otherParticleIndex++) {
				Particle otherParticle = particles[otherParticleIndex];

				if (particle.m_ID == otherParticle.m_ID) {
					continue;
				}

				float distance = sqrt(CudaMath::squared_distance(particle.m_PredictedPosition, otherParticle.m_PredictedPosition));

				GpuVector2D dir = distance < particleRadius ? GpuVector2D::getRandomDirection() : (GpuVector2D(otherParticle.m_PredictedPosition) - GpuVector2D(particle.m_PredictedPosition)) / distance;

				float slope = CudaMath::smoothingKernelDerivative(particleRadiusOfRepel, distance);

				float density = otherParticle.m_Density;

				/*if (density == 0) {
					printf("density: %f, index: %d\n", density, index);
				}*/

				float sharedPressure = CudaMath::calculateSharedPressure(density, otherParticle.m_Density);

				pressureForce += -sharedPressure * dir * slope * mass / density;
			}
		}
	}
	/*printf("pressureForceX: %f , pressureForceY: %f, index: %d\n", pressureForce.X, pressureForce.Y, index);*/
	return pressureForce;
}

__device__ void updateParticleFutureVelocities(int index, Particle* particles, int praticlesSize,
	int particleRadiusOfRepel, int particleRadius, Range* lengths, size_t interactionMatrixRows,
	size_t interactionMatrixCols, double dt)
{

	Particle particle = particles[index];

	if (particle.m_Density == 0) {
		return;
	}

	GpuVector2D pressureForce = calculatePressureForce2(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		lengths, interactionMatrixRows, interactionMatrixCols);

	//pressureForce = GpuVector2D(300, 300);
	/*if (isnan(pressureForce.X) || isnan(pressureForce.Y)) {
		printf("pressureForceX: %f , pressureForceY: %f, index: %d\n", pressureForce.X, pressureForce.Y, index);
	}*/
	GpuVector2D pressureAcceleration = pressureForce / particle.m_Density;

	//Vector2D viscosityForce = calculateViscosityForce(particle);
	GpuVector2D viscosityForce = GpuVector2D();

	GpuVector2D futureVelocity = GpuVector2D(particle.m_Velocity) + pressureAcceleration * dt + viscosityForce * dt;

	//printf("index: %d, futureVelocity: %f %f \n", index, futureVelocity.X, futureVelocity.Y);

	particles[index].m_FutureVelocity.X = futureVelocity.X;
	particles[index].m_FutureVelocity.Y = futureVelocity.Y;


	particles[index].m_Velocity = particles[index].m_FutureVelocity;
	updateParticle(index, particles, dt);
	particles[index].m_PredictedPosition = particles[index].m_Position;
}

__device__ void updateCollisions(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, Surface2D* obstacles, int obstaclesSize)
{


	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	int row = point.Y / particleRadiusOfRepel;
	int col = point.X / particleRadiusOfRepel;

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


	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if (row + i < 0 || row + i >= interactionMatrixRows || col + j < 0 || col + j >= interactionMatrixCols) {
				continue;
			}

			int lengthIndex = (row + i) * interactionMatrixCols + col + j;

			for (int otherParticleIndex = lengths[lengthIndex].start;
				otherParticleIndex < lengths[lengthIndex].end; otherParticleIndex++) {

				Particle otherParticle = particles[otherParticleIndex];

				if (particle.m_ID == otherParticle.m_ID) {
					continue;
				}

				if (CudaMath::squared_distance(particle.m_Position, otherParticle.m_Position) <=
					(particleRadius * particleRadius) * 4) {

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
	}
}

__global__ void specialUpdateKernel(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, Surface2D* obstacles, int obstaclesSize, double dt) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//printf("important index: %d \n", index);

	if (index >= praticlesSize) {
		return;
	}

	//printf("index: %d \n", index);

	Particle particle = particles[index];

	GpuVector2D newPredictedPosition = GpuVector2D(particles[index].m_Position) +
		GpuVector2D(particles[index].m_Velocity) * dt * HOW_FAR_INTO_THE_FUTURE;

	//printf("index: %d, newPredictedPosition: %f %f \n", index, newPredictedPosition.X, newPredictedPosition.Y);

	particles[index].m_PredictedPosition.X = newPredictedPosition.X;
	particles[index].m_PredictedPosition.Y = newPredictedPosition.Y;

	// Synchronize all threads in the block
	atomicAdd(&counterPredictedPositionsDone, 1);

	while (counterPredictedPositionsDone < praticlesSize) {
		continue;
	}
	// calculate densities

	updateParticleDensities(index, particles, praticlesSize, particleRadiusOfRepel, lengths,
		interactionMatrixRows, interactionMatrixCols);

	//printf("index: %d, density: %f \n", index, particles[index].m_Density);

	// calculate densities

	atomicAdd(&counterDensitiesDone, 1);

	while (counterDensitiesDone < praticlesSize) {
		continue;
	}

	// calculate future velocities

	updateParticleFutureVelocities(index, particles, praticlesSize, particleRadiusOfRepel,
		particleRadius, lengths, interactionMatrixRows, interactionMatrixCols, dt);

	// calculate future velocities

	atomicAdd(&counterFutureVelocitiesDone, 1);

	while (counterFutureVelocitiesDone < praticlesSize) {
		continue;
	}

	// check collisions

	updateCollisions(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		particleRepulsionForce, lengths, interactionMatrixRows, interactionMatrixCols, obstacles, obstaclesSize);

	// check collisions


}

__global__ void resetGlobalCounter() {
	counterDensitiesDone = 0;
	counterPredictedPositionsDone = 0;
	counterFutureVelocitiesDone = 0;
}

void GpuAllocateInteractionMatrix(InteractionMatrixClass* interactionMatrix) {


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

	int blockSize2 = maxThreadsPerBlock;
	int numBlocks2 = particles.size();

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	// Launch CUDA kernel
	updateParticleDensitiesKernel << <numBlocks, blockSize >> > (gpuParticles, particles.size(), particleRadiusOfRepel,
		deviceInteractionMatrixParticles, deviceLengths, interactionMatrixRows, interactionMatrixCols,
		maxParticlesInInteractionMatrixCell);

	/*updateParticleDensitiesKernel2 << <numBlocks2, blockSize2 >> > (gpuParticles, particles.size(), particleRadiusOfRepel,
		deviceInteractionMatrixParticles, deviceLengths, interactionMatrixRows, interactionMatrixCols,
		maxParticlesInInteractionMatrixCell);*/

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
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles,
	double dt)
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

__device__ Range divideEtImpera(Particle* particles, int left, int right, int particlesSize,
	int particleRadiusOfRepel, int expectedPosition, int interactionMatrixCols) {

	do {
		if (left >= right) {
			return Range{ 0,0 };
		}

		int mid = left + (right - left) / 2;

		int row = particles[mid].m_Position.Y / particleRadiusOfRepel;
		int col = particles[mid].m_Position.X / particleRadiusOfRepel;

		int position = row * interactionMatrixCols + col;

		if (position == expectedPosition) {
			Range range{ 0,0 };

			for (int index = mid; index >= 0; index--) {
				int currentRow = particles[index].m_Position.Y / particleRadiusOfRepel;
				int currentCol = particles[index].m_Position.X / particleRadiusOfRepel;

				int currentPosition = currentRow * interactionMatrixCols + currentCol;

				if (currentPosition != expectedPosition) {
					range.start = index + 1;
					break;
				}
			}

			range.end = particlesSize;
			for (int index = mid; index < particlesSize; index++) {
				int currentRow = particles[index].m_Position.Y / particleRadiusOfRepel;
				int currentCol = particles[index].m_Position.X / particleRadiusOfRepel;

				int currentPosition = currentRow * interactionMatrixCols + currentCol;

				if (currentPosition != expectedPosition) {
					range.end = index;
					break;
				}
			}

			return range;
		}

		if (position < expectedPosition) {
			left = mid + 1;
		}
		else {
			right = mid - 1;
		}
	} while (true);
}

__global__ void setLengths(Particle* particles, int particlesSize, int particleRadiusOfRepel, Range* lengths, int interactionMatrixRows, int interactionMatrixCols) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= interactionMatrixRows * interactionMatrixCols) {
		return;
	}

	//printf("index: %d, start: %d, end: %d \n", index, lengths[index].start, lengths[index].end);
	lengths[index] = divideEtImpera(particles, 0, particlesSize - 1, particlesSize,
		particleRadiusOfRepel, index, interactionMatrixCols);

}

//GPU Kernel Implementation of Bitonic Sort
__global__ void bitonicSortGPU(Particle* arr, int j, int k, int particleRadiusOfRepel)
{
	unsigned int i, ij;

	i = threadIdx.x + blockDim.x * blockIdx.x;

	ij = i ^ j;

	if (ij > i)
	{
		int rowA = arr[i].m_Position.Y / particleRadiusOfRepel;
		int colA = arr[i].m_Position.X / particleRadiusOfRepel;

		int rowB = arr[ij].m_Position.Y / particleRadiusOfRepel;
		int colB = arr[ij].m_Position.X / particleRadiusOfRepel;

		bool lower;

		if (rowA == rowB) {
			lower = colA < colB;
		}
		else {
			lower = rowA < rowB;
		}

		if ((i & k) == 0)
		{
			if (!lower)
			{
				Particle temp = arr[i];
				arr[i] = arr[ij];
				arr[ij] = temp;
			}
		}
		else
		{
			if (lower)
			{
				Particle temp = arr[i];
				arr[i] = arr[ij];
				arr[ij] = temp;
			}
		}
	}
}

void GpuAllocate(std::vector<Particle>& particles, std::vector<Surface2D>& obstacles, int interactionMatrixSize) {

	cudaError_t cudaStatus;

	// Allocate memory on GPU
	cudaStatus = cudaMalloc(&deviceParticles, particles.size() * sizeof(Particle));
	cudaStatus = cudaMalloc(&deviceObstacles, obstacles.size() * sizeof(Surface2D));

	// Copy data from CPU to GPU
	cudaStatus = cudaMemcpy(deviceParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(deviceObstacles, obstacles.data(), obstacles.size() * sizeof(Surface2D), cudaMemcpyHostToDevice);


	// Allocate memory on GPU
	cudaStatus = cudaMalloc(&lengths, interactionMatrixSize * sizeof(Range));

	Range* hostLengths = new Range[interactionMatrixSize]{ Range{0,0} };
	cudaStatus = cudaMemcpy(lengths, hostLengths, interactionMatrixSize * sizeof(Range), cudaMemcpyHostToDevice);

	delete[] hostLengths;
}

void GpuFree() {
	// Free GPU memory
	cudaFree(deviceParticles);
	cudaFree(deviceObstacles);
	cudaFree(lengths);
}

__global__ void demo(Particle* particles, int particlesSize) {
	printf("particlesSize: %d \n", particlesSize);
	/*for (int i = 0; i < particlesSize; i++) {
		printf("index: %d, position: %f %f \n", i, particles[i].m_Position.X, particles[i].m_Position.Y);
	}*/
}

void GpuUpdateParticles(std::vector<Particle>& particles, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles,
	double dt, size_t interactionMatrixRows, size_t interactionMatrixCols,
	InteractionMatrixClass* interactionMatrix) {

	interactionMatrixSize = interactionMatrixRows * interactionMatrixCols;

	//cudaMalloc(&deviceParticles, particles.size() * sizeof(Particle));
	//cudaMemcpy(deviceParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	//Set number of threads and blocks for kernel calls
	int threadsPerBlock = maxThreadsPerBlock;
	int blocksPerGrid = (particles.size() + threadsPerBlock - 1) / threadsPerBlock;
	int k, j;

	//demo << <1, 1 >> > (deviceParticles, particles.size());
	//cudaDeviceSynchronize();

	// Bitonic Sort
	for (k = 2; k <= particles.size(); k <<= 1)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			bitonicSortGPU << <blocksPerGrid, threadsPerBlock >> > (deviceParticles, j, k, particleRadiusOfRepel);
		}
	}
	cudaDeviceSynchronize();

	//printf("\n\n\n 111111111111111111111111111111111 \n\n\n");

	//demo << <1, 1 >> > (deviceParticles, particles.size());
	//cudaDeviceSynchronize();

	int blockSize = (interactionMatrixSize < maxThreadsPerBlock) ? interactionMatrixSize : maxThreadsPerBlock;
	int numBlocks = (interactionMatrixSize + blockSize - 1) / blockSize;

	// Launch CUDA kernel for setting lengths
	setLengths << < numBlocks, blockSize >> > (deviceParticles, particles.size(), particleRadiusOfRepel,
		lengths, interactionMatrixRows, interactionMatrixCols);

	// Wait for kernel to finish
	//cudaDeviceSynchronize();

	//printf("\n\n\n 2222222222222222222222222222222222 \n\n\n");

	//demo << <1, 1 >> > (deviceParticles, particles.size());
	//cudaDeviceSynchronize();

	resetGlobalCounter << <1, 1 >> > ();

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	blockSize = (particles.size() < maxThreadsPerBlock) ? particles.size() : maxThreadsPerBlock;
	numBlocks = (particles.size() + blockSize - 1) / blockSize;

	//printf("\n\n\n 33333333333333333333333333333333 \n\n\n");

	//demo << <1, 1 >> > (deviceParticles, particles.size());
	//cudaDeviceSynchronize();

	// Launch CUDA kernel for updating particles
	specialUpdateKernel << <numBlocks, blockSize >> > (deviceParticles, particles.size(), particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, lengths, interactionMatrixRows,
		interactionMatrixCols, deviceObstacles, obstacles.size(), dt);

	// Wait for kernel to finish
	cudaDeviceSynchronize();


	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, deviceParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
	}

	// Free output
	delete[] output;
}
