#ifndef GpuParallel_h
#define GpuParallel_h

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.hpp"

#include <vector>
#include "InteractionMatrixClass.hpp"

#include <curand_kernel.h>

struct GpuVector2D {
	

	__device__ GpuVector2D() {
		X = 0;
		Y = 0;
	}

	__device__ GpuVector2D(float x, float y) {
		X = x;
		Y = y;
	}

	__device__ GpuVector2D(Vector2D position) {
		X = position.X;
		Y = position.Y;
	}

	// Function to calculate the magnitude of a 2D vector (x, y)
	__device__ float getMagnitude() const {
		return std::sqrt(X * X + Y * Y);
	}

	__device__ float getMagnitudeSquared() const {
		return X * X + Y * Y;
	}

	__device__ static GpuVector2D getRandomDirection() {
		// Create a random number generator
		curandStatePhilox4_32_10_t state;
		curand_init(1234, 0, 0, &state);

		// Generate a random angle
		float angle = curand_uniform(&state) * 2 * constants::m_PI;

		// Calculate the x and y components of the direction vector
		float x = std::cos(angle);
		float y = std::sin(angle);

		// Create and output the direction vector
		GpuVector2D direction = { x, y };

		return direction;
	}

	// add 2 vectors
	__device__ GpuVector2D operator+(const GpuVector2D& other) const {
		return GpuVector2D(X + other.X, Y + other.Y);
	}

	// multiply by a scalar
	__device__ GpuVector2D operator*(float scalar) const {
		return GpuVector2D(X * scalar, Y * scalar);
	}

	// dot product
	__device__ float operator*(const GpuVector2D& other) const {
		return X * other.X + Y * other.Y;
	}

	// add 1 vector to current vector
	__device__ GpuVector2D& operator+=(const GpuVector2D& other) {
		X += other.X;
		Y += other.Y;
		return *this;
	}

	// substract 2 vectors
	__device__ GpuVector2D operator-(const GpuVector2D& other) const {
		return GpuVector2D(X - other.X, Y - other.Y);
	}

	// subtract a vector from the current vector
	__device__ GpuVector2D& operator-=(const GpuVector2D& other) {
		X -= other.X;
		Y -= other.Y;
		return *this;
	}

	// divide by a scalar
	__device__ GpuVector2D operator/(float scalar) const {
		return GpuVector2D(X / scalar, Y / scalar);
	}

	// float times vector
	__device__ friend GpuVector2D operator*(float scalar, const GpuVector2D& vector) {
		return vector * scalar;
	}

	// minus vector
	__device__ GpuVector2D operator-() const {
		return GpuVector2D(-X, -Y);
	}

	float X;
	float Y;
};

void GpuParallelUpdateParticleDensities(std::vector<Particle>& particles, int particleRadiusOfRepel);

void GpuParallelCalculateFutureVelocities(std::vector<Particle>& particles, int particleRadiusOfRepel, int particleRadius, double dt);

void GpuAllocateInteractionMatrix(InteractionMatrixClass* interactionMatrix);

void GpuFreeInteractionMatrix();

#endif