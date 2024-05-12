#include "GpuParallel.cuh"
#include "CudaMath.cuh"

#include <chrono>
#include "Phisics.hpp"

#include <cuda_runtime.h>
#include <random>

constexpr float HOW_FAR_INTO_THE_FUTURE = 2.5f;

constexpr int maxThreadsPerBlock = 512;

constexpr float viscosityStrength = 0.1f;

struct Range {
	int start;
	int end;
};


Particle* deviceParticles;

Range* lengths;

Surface2D* deviceObstacles;

ConsumerPipe* deviceConsumerPipes;
GeneratorPipe* deviceGeneratorPipes;

int maxParticles = 0;

size_t consumerPipesSize;
size_t generatorPipesSize;

Particle* deviceInteractionMatrixParticles;

int* deviceLengths;
int interactionMatrixSize;

size_t maxParticlesInInteractionMatrixCell;
size_t interactionMatrixRows;
size_t interactionMatrixCols;

__device__ int counterDensitiesDone = 0;
__device__ int counterPredictedPositionsDone = 0;
__device__ int counterFutureVelocitiesDone = 0;

__device__ void updateParticle(int index, Particle* particle, double dt) {
	if (dt == 0) {
		return;
	}

	float leapFrog = 1.0f;
	//float leapFrog = 0.99f;

	particle[index].m_LastSafePosition = particle[index].m_Position;

	GpuVector2D newVelocity{ 0,0 };

	newVelocity = GpuVector2D(particle[index].m_Velocity) + GpuVector2D(0.0f, GRAVITY) * dt;

	newVelocity += GpuVector2D(particle[index].m_TemporaryVelocity);

	particle[index].m_Velocity.X = newVelocity.X * leapFrog;
	particle[index].m_Velocity.Y = newVelocity.Y * leapFrog;

	particle[index].m_TemporaryVelocity.X = 0;
	particle[index].m_TemporaryVelocity.Y = 0;

	particle[index].m_Position.X += particle[index].m_Velocity.X * dt;
	particle[index].m_Position.Y += particle[index].m_Velocity.Y * dt;
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
				if (!otherParticle.m_Exists) {
					continue;
				}
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

__device__ GpuVector2D calculatePressureForce(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
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

				pressureForce += -sharedPressure * dir * slope * mass /*/ density*/;
			}
		}
	}
	/*printf("pressureForceX: %f , pressureForceY: %f, index: %d\n", pressureForce.X, pressureForce.Y, index);*/
	return pressureForce;
}


__device__ GpuVector2D calculateViscosityForce(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, Range* lengths, int interactionMatrixRows, int interactionMatrixCols) {

	Particle particle = particles[index];

	GpuVector2D viscosityForce = GpuVector2D();
	Vector2D point = particle.m_PredictedPosition;

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
				float influence = CudaMath::viscositySmoothingKernel(particleRadiusOfRepel, distance);

				viscosityForce += (GpuVector2D(otherParticle.m_Velocity) - GpuVector2D(particle.m_Velocity)) * influence;
			}
		}
	}

	return viscosityForce * viscosityStrength;
}

__device__ void updateParticleFutureVelocities(int index, Particle* particles, int praticlesSize,
	int particleRadiusOfRepel, int particleRadius, Range* lengths, size_t interactionMatrixRows,
	size_t interactionMatrixCols, double dt)
{

	Particle particle = particles[index];

	if (particle.m_Density == 0) {
		return;
	}

	GpuVector2D pressureForce = calculatePressureForce(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		lengths, interactionMatrixRows, interactionMatrixCols);

	//pressureForce = GpuVector2D(300, 300);
	/*if (isnan(pressureForce.X) || isnan(pressureForce.Y)) {
		printf("pressureForceX: %f , pressureForceY: %f, index: %d\n", pressureForce.X, pressureForce.Y, index);
	}*/
	GpuVector2D pressureAcceleration = pressureForce / particle.m_Density;

	GpuVector2D viscosityForce = calculateViscosityForce(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		lengths, interactionMatrixRows, interactionMatrixCols);
	//GpuVector2D viscosityForce = GpuVector2D();

	GpuVector2D futureVelocity = GpuVector2D(particle.m_Velocity) + pressureAcceleration * dt + viscosityForce * dt;

	//printf("index: %d, futureVelocity: %f %f \n", index, futureVelocity.X, futureVelocity.Y);

	particles[index].m_FutureVelocity.X = futureVelocity.X;
	particles[index].m_FutureVelocity.Y = futureVelocity.Y;


	particles[index].m_Velocity = particles[index].m_FutureVelocity;
	updateParticle(index, particles, dt);
	particles[index].m_PredictedPosition = particles[index].m_Position;
}

__device__ void updateCollisionsBetweenParticlesAndObstacles(int particleIndex, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, int obstacleIndex, Surface2D* obstacles, int obstaclesSize)
{

	Particle particle = particles[particleIndex];

	Vector2D point = particle.m_PredictedPosition;

	int row = point.Y / particleRadiusOfRepel;
	int col = point.X / particleRadiusOfRepel;

	Surface2D obstacle = obstacles[obstacleIndex];

	if (CudaMath::check_line_segment_circle_intersection(obstacle.Point1, obstacle.Point2,
		particle.m_Position, particleRadius)) {

		//particle->m_Velocity = reflectionVector * 0.1f;
		particles[particleIndex].m_Velocity.X = 0;
		particles[particleIndex].m_Velocity.Y = 0;

		/*particle.m_Velocity.Y = -particle.m_Velocity.Y;*/

		particles[particleIndex].m_Position = particle.m_LastSafePosition;

		return;
	}

	if (CudaMath::doIntersect(obstacle.Point1, obstacle.Point2, particle.m_Position, particle.m_LastSafePosition)) {
		//particle->m_Velocity = reflectionVector * 0.1f;
		particles[particleIndex].m_Velocity.X = 0;
		particles[particleIndex].m_Velocity.Y = 0;

		/*particle.m_Velocity.Y = -particle.m_Velocity.Y;*/

		particles[particleIndex].m_Position = particle.m_LastSafePosition;

	}
}

__device__ void updateCollisionsBetweenParticlesAndParticles(int index, Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols)
{


	Particle particle = particles[index];

	Vector2D point = particle.m_PredictedPosition;

	int row = point.Y / particleRadiusOfRepel;
	int col = point.X / particleRadiusOfRepel;


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

				if (!otherParticle.m_Exists) {
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

					//particles[index].m_Position = particles[index].m_LastSafePosition;

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

	printf("important index: %d \n", index);

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

	/*updateCollisions(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		particleRepulsionForce, lengths, interactionMatrixRows, interactionMatrixCols, obstacles, obstaclesSize);*/

		// check collisions


}

__global__ void specialUpdatePredictedPositions(Particle* particles, int praticlesSize, double dt) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	Particle particle = particles[index];

	GpuVector2D newPredictedPosition = GpuVector2D(particles[index].m_Position) +
		GpuVector2D(particles[index].m_Velocity) * dt * HOW_FAR_INTO_THE_FUTURE;

	particles[index].m_PredictedPosition.X = newPredictedPosition.X;
	particles[index].m_PredictedPosition.Y = newPredictedPosition.Y;
}

__global__ void specialUpdateDensities(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, Surface2D* obstacles, int obstaclesSize, double dt) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	updateParticleDensities(index, particles, praticlesSize, particleRadiusOfRepel, lengths,
		interactionMatrixRows, interactionMatrixCols);
}

__global__ void specialUpdateFutureVelocities(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, Surface2D* obstacles, int obstaclesSize, double dt) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	updateParticleFutureVelocities(index, particles, praticlesSize, particleRadiusOfRepel,
		particleRadius, lengths, interactionMatrixRows, interactionMatrixCols, dt);
}

__global__ void specialUpdateCollisionsBetweenParticlesAndObstacles(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, Surface2D* obstacles, int obstaclesSize, double dt) {

	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= praticlesSize) {
		return;
	}

	int obstacleIndex = blockIdx.y;

	if (obstacleIndex >= obstaclesSize) {
		return;
	}

	updateCollisionsBetweenParticlesAndObstacles(particleIndex, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		particleRepulsionForce, lengths, interactionMatrixRows, interactionMatrixCols, obstacleIndex, obstacles, obstaclesSize);
}

__global__ void specialUpdateCollisionsBetweenParticlesAndParticles(Particle* particles, int praticlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, Range* lengths, int interactionMatrixRows,
	int interactionMatrixCols, double dt) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= praticlesSize) {
		return;
	}

	updateCollisionsBetweenParticlesAndParticles(index, particles, praticlesSize, particleRadiusOfRepel, particleRadius,
		particleRepulsionForce, lengths, interactionMatrixRows, interactionMatrixCols);
}

__global__ void resetGlobalCounter() {
	counterDensitiesDone = 0;
	counterPredictedPositionsDone = 0;
	counterFutureVelocitiesDone = 0;
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

		if (!arr[ij].m_Exists) {
			lower = true;
		}
		else {
			if (!arr[i].m_Exists) {
				lower = false;
			}
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

void GpuAllocate(std::vector<Particle>& particles, std::vector<Surface2D>& obstacles, int interactionMatrixSize, std::vector<ConsumerPipe>& consumerPipes,
	std::vector<GeneratorPipe>& generatorPipes) {

	cudaError_t cudaStatus;

	// Get max particles of all generator pipes
	for (int i = 0; i < generatorPipes.size(); i++) {
		maxParticles = maxParticles < generatorPipes[i].m_ParticlesPerCycle ? generatorPipes[i].m_ParticlesPerCycle : maxParticles;
	}

	consumerPipesSize = consumerPipes.size();
	generatorPipesSize = generatorPipes.size();

	// Allocate memory on GPU
	cudaStatus = cudaMalloc(&deviceParticles, particles.size() * sizeof(Particle));
	cudaStatus = cudaMalloc(&deviceObstacles, obstacles.size() * sizeof(Surface2D));
	cudaStatus = cudaMalloc(&deviceConsumerPipes, consumerPipes.size() * sizeof(ConsumerPipe));
	cudaStatus = cudaMalloc(&deviceGeneratorPipes, generatorPipes.size() * sizeof(GeneratorPipe));

	// Copy data from CPU to GPU
	cudaStatus = cudaMemcpy(deviceParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(deviceObstacles, obstacles.data(), obstacles.size() * sizeof(Surface2D), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(deviceConsumerPipes, consumerPipes.data(), consumerPipes.size() * sizeof(ConsumerPipe), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(deviceGeneratorPipes, generatorPipes.data(), generatorPipes.size() * sizeof(GeneratorPipe), cudaMemcpyHostToDevice);


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
	cudaFree(deviceConsumerPipes);
	cudaFree(deviceGeneratorPipes);
	cudaFree(lengths);
}

void UpdateParticlesHelper(std::vector<Particle>& particles, int particlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles,
	double dt, size_t interactionMatrixRows, size_t interactionMatrixCols) {

	int blockSize = (particlesSize < maxThreadsPerBlock) ? particlesSize : maxThreadsPerBlock;
	int numBlocks = (particlesSize + blockSize - 1) / blockSize;

	specialUpdatePredictedPositions << <numBlocks, blockSize >> > (deviceParticles, particlesSize, dt);

	cudaDeviceSynchronize();

	specialUpdateDensities << <numBlocks, blockSize >> > (deviceParticles, particlesSize, particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, lengths, interactionMatrixRows,
		interactionMatrixCols, deviceObstacles, obstacles.size(), dt);

	cudaDeviceSynchronize();

	specialUpdateFutureVelocities << <numBlocks, blockSize >> > (deviceParticles, particlesSize, particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, lengths, interactionMatrixRows,
		interactionMatrixCols, deviceObstacles, obstacles.size(), dt);

	cudaDeviceSynchronize();

	dim3 gridDim(numBlocks, obstacles.size()); // 4x4 blocks

	specialUpdateCollisionsBetweenParticlesAndObstacles << <gridDim, blockSize >> > (deviceParticles, particlesSize, particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, lengths, interactionMatrixRows,
		interactionMatrixCols, deviceObstacles, obstacles.size(), dt);

	specialUpdateCollisionsBetweenParticlesAndParticles << <numBlocks, blockSize >> > (deviceParticles, particlesSize, particleRadiusOfRepel,
		particleRadius, particleRepulsionForce, lengths, interactionMatrixRows, interactionMatrixCols, dt);

	// Wait for kernel to finish
	cudaDeviceSynchronize();
}

__global__ void specialUpdateConsumerPipes(Particle* particles, int particlesSize, int particleRadius, ConsumerPipe* consumerPipes, size_t consumerPipesSize) {
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (particleIndex >= particlesSize) {
		return;
	}

	int pipeIndex = blockIdx.y;

	if (pipeIndex >= consumerPipesSize) {
		return;
	}

	Particle particle = particles[particleIndex];

	ConsumerPipe pipe = consumerPipes[pipeIndex];

	GpuVector2D direction = GpuVector2D(particle.m_Position) - GpuVector2D(pipe.m_Position);

	if (direction.getMagnitude() == 0) {
		direction = GpuVector2D::getRandomDirection();
	}

	float distance = direction.getMagnitude();
	if (distance <= pipe.m_InteractionRadius + particleRadius) {
		particles[particleIndex].m_Exists = false;
	}

	/*if (particlesToRemove.size() >= m_ParticlesPerCycle) {
		break;
	}*/

}

__global__ void specialUpdateGeneratorPipes(Particle* particles, int particlesSize, int particleRadius, GeneratorPipe* generatorPipes,
	size_t generatorPipesSize, int seed) {
	int index = threadIdx.x;

	int pipeIndex = blockIdx.x;

	if (index >= generatorPipes[pipeIndex].m_ParticlesPerCycle) {
		return;
	}

	if (pipeIndex >= generatorPipesSize) {
		return;
	}

	int particleIndex = particlesSize - (pipeIndex * blockDim.x + index + 1);

	curandStatePhilox4_32_10_t state;
	curand_init(seed, particleIndex, 0, &state);

	// Generate a random number between -1 and 1
	float random = curand_uniform(&state) * 2 - 1;
	float random2 = curand_uniform(&state);

	int posX = generatorPipes[pipeIndex].m_Position.X + random * generatorPipes[pipeIndex].m_InteractionRadius;
	int posY = generatorPipes[pipeIndex].m_Position.Y + random2 * generatorPipes[pipeIndex].m_InteractionRadius;

	particles[particleIndex].m_Position.X = posX;
	particles[particleIndex].m_Position.Y = posY;

	particles[particleIndex].m_Exists = true;

	//printf("particleIndex: %d, posX: %d, posY: %d \n", particleIndex, posX, posY);

}

void UpdatePipes(int particlesSize, int particleRadius) {

	int blockSize = (particlesSize < maxThreadsPerBlock) ? particlesSize : maxThreadsPerBlock;
	int numBlocks = (particlesSize + blockSize - 1) / blockSize;

	dim3 gridDim(numBlocks, consumerPipesSize); // 4x4 blocks

	specialUpdateConsumerPipes << <gridDim, blockSize >> > (deviceParticles, particlesSize,
		particleRadius, deviceConsumerPipes, consumerPipesSize);

	numBlocks = generatorPipesSize;
	blockSize = maxParticles;

	//gridDim = dim3(numBlocks, generatorPipesSize); // 4x4 blocks

	int particleArraySize = Math::nextPowerOf2(particlesSize);

	// Generate random seed
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-1.0, 1.0);

	int seed = dis(gen) * 1000;

	specialUpdateGeneratorPipes << <numBlocks, blockSize >> > (deviceParticles, particleArraySize,
		particleRadius, deviceGeneratorPipes, generatorPipesSize, seed);

	// Wait for kernel to finish
	cudaDeviceSynchronize();


}

void GpuUpdateParticles(std::vector<Particle>& particles, int& particlesSize, int particleRadiusOfRepel,
	int particleRadius, float particleRepulsionForce, std::vector<Surface2D>& obstacles,
	double dt, size_t interactionMatrixRows, size_t interactionMatrixCols) {

	interactionMatrixSize = interactionMatrixRows * interactionMatrixCols;

	//Set number of threads and blocks for kernel calls
	int threadsPerBlock = maxThreadsPerBlock;
	int blocksPerGrid = (particles.size() + threadsPerBlock - 1) / threadsPerBlock;
	int k, j;

	// Bitonic Sort
	for (k = 2; k <= particles.size(); k <<= 1)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			bitonicSortGPU << <blocksPerGrid, threadsPerBlock >> > (deviceParticles, j, k, particleRadiusOfRepel);
		}
	}

	int blockSize = (interactionMatrixSize < maxThreadsPerBlock) ? interactionMatrixSize : maxThreadsPerBlock;
	int numBlocks = (interactionMatrixSize + blockSize - 1) / blockSize;

	cudaDeviceSynchronize();

	// Launch CUDA kernel for setting lengths
	setLengths << < numBlocks, blockSize >> > (deviceParticles, particlesSize, particleRadiusOfRepel,
		lengths, interactionMatrixRows, interactionMatrixCols);

	resetGlobalCounter << <1, 1 >> > ();

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Launch CUDA kernel for updating particles
	UpdateParticlesHelper(particles, particlesSize, particleRadiusOfRepel, particleRadius, particleRepulsionForce, obstacles, dt, interactionMatrixRows, interactionMatrixCols);


	// Update pipes
	UpdatePipes(particlesSize, particleRadius);

	Particle* output = new Particle[particles.size()];

	cudaMemcpy(output, deviceParticles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

	particlesSize = 0;

	for (int i = 0; i < particles.size(); i++) {
		particles[i] = output[i];
		if (particles[i].m_Exists) {
			particlesSize++;
		}
	}

	// Free output
	delete[] output;
}
