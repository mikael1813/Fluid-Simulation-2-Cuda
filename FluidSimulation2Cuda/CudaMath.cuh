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


} // namespace CudaMath