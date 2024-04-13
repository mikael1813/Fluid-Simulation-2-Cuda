#include "Phisics.hpp"


namespace CudaMath {


	// Function to compute the squared distance between two points
	__device__ float squared_distance(const Vector2D& p1, const Vector2D& p2) {
		float dx = p1.X - p2.X;
		float dy = p1.Y - p2.Y;
		return dx * dx + dy * dy;
	}

	__device__ float smoothingKernel(float radius, float distance) {
		if (distance >= radius) {
			return 0.0f;
		}

		float x = (radius - distance) / radius;
		return x * x;
	}

	__device__ float smoothingKernelDerivative(float radius, float distance) {
		if (distance >= radius) {
			return 0.0f;
		}
		float x = (radius - distance) / radius;
		return 2 * x;
	}

	__device__ float convertDensityToPressure(float density) {
		const float targetDensity = 0.5f;
		//const float pressureConstant = 10.0f;
		const float pressureConstant = 30.0f;

		float densityError = density - targetDensity;
		float pressure = pressureConstant * densityError;
		return pressure;
	}

	__device__ float calculateSharedPressure(float density1, float density2) {
		float pressure1 = convertDensityToPressure(density1);
		float pressure2 = convertDensityToPressure(density2);
		return (pressure1 + pressure2) / 2;
	}


} // namespace CudaMath