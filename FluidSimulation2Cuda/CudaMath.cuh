#include "Phisics.hpp"

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

namespace CudaMath {


	// Function to compute the squared distance between two points
	__device__ float squared_distance(const Vector2D& p1, const Vector2D& p2) {
		float dx = p1.X - p2.X;
		float dy = p1.Y - p2.Y;
		return dx * dx + dy * dy;
	}

	// Function to compute the squared distance between two points
	__device__ float squared_distance(const Vector2D& p1, const GpuVector2D& p2) {
		float dx = p1.X - p2.X;
		float dy = p1.Y - p2.Y;
		return dx * dx + dy * dy;
	}

	// Check if the line segment AB intersects the circle with center C and radius R
	__device__ bool check_line_segment_circle_intersection(const Vector2D& A, const Vector2D& B, const Vector2D& C, float radius) {
		// Compute squared distances
		float dist_AB_squared = squared_distance(A, B);
		float dist_AC_squared = squared_distance(A, C);
		float dist_BC_squared = squared_distance(B, C);

		// Check if any of the endpoints (A or B) is inside the circle
		if (dist_AC_squared <= radius * radius || dist_BC_squared <= radius * radius) {
			return true;
		}

		// Check if the line segment intersects the circle
		float dot_product = (C.X - A.X) * (B.X - A.X) + (C.Y - A.Y) * (B.Y - A.Y);
		if (dot_product < 0 || dot_product > dist_AB_squared) {
			return false; // Closest point to C is outside the line segment AB
		}

		// Compute the closest point P on the line segment AB to C
		GpuVector2D P;
		P.X = A.X + (B.X - A.X) * dot_product / dist_AB_squared;
		P.Y = A.Y + (B.Y - A.Y) * dot_product / dist_AB_squared;

		// Check if the distance from P to C is less than or equal to the radius
		float dist_CP_squared = squared_distance(C, P);
		return dist_CP_squared <= radius * radius;
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
		const float targetDensity = 3.0f;
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