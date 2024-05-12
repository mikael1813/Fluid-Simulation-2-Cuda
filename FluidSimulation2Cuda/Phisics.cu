#include "Phisics.cuh"
#include <random>

#include <curand_kernel.h>

__device__ Vector2D Vector2D::getRandomDirection() {

	//// Seed the random number generator
	//std::random_device rd;
	//std::mt19937 gen(rd());

	//// Create a uniform distribution for the angle (in radians)
	//std::uniform_real_distribution<float> dist(0, 2 * constants::m_PI); // Range: [0, 2 * pi]

	//// Generate a random angle
	//float angle = dist(gen);

	//// Calculate the x and y components of the direction vector
	//float x = std::cos(angle);
	//float y = std::sin(angle);

	//// Create and output the direction vector
	//Vector2D direction = { x, y };

	//return direction;

	curandStatePhilox4_32_10_t state;
	curand_init(1234, 0, 0, &state);

	// Generate a random angle
	float angle = curand_uniform(&state) * 2 * constants::m_PI;

	// Calculate the x and y components of the direction vector
	float x = std::cos(angle);
	float y = std::sin(angle);

	// Create and output the direction vector
	Vector2D direction = { x, y };

	return direction;
}

// Function to calculate the slope of a line given two points
__host__ __device__ double Math::calculateSlope(Vector2D a, Vector2D b) {
	// Ensure x2 is not equal to x1 to avoid division by zero
	if (a.X == b.X) {
		/*std::cerr << "Error: Division by zero (x2 - x1 = 0)" << std::endl;*/
		return INFINITY;
	}

	// Calculate the slope using the formula (y2 - y1) / (x2 - x1)
	return (b.Y - a.Y) / (b.X - a.X);
}

// Function to calculate the normal vector given the slope of the surface line
__host__ __device__ Vector2D Math::calculateNormalVector(double surfaceLineSlope) {
	// Calculate the slope of the perpendicular line
	double perpendicularLineSlope;

	if (surfaceLineSlope == 0.0) {
		// Handle the case when the surface line is horizontal (slope is 0)
		perpendicularLineSlope = INFINITY; // Treat the perpendicular line slope as infinity
	}
	else {
		// Calculate the slope of the perpendicular line (negative reciprocal)
		perpendicularLineSlope = -1.0 / surfaceLineSlope;
	}

	// The normal vector is represented by the coefficients (1, m), where m is the perpendicular line slope
	double normalX = 1.0;
	double normalY = perpendicularLineSlope;

	// Calculate the magnitude of the normal vector
	double magnitude = sqrt(normalX * normalX + normalY * normalY);

	// Normalize the components to obtain the direction of the normal vector
	normalX /= magnitude;
	normalY /= magnitude;

	/*if (abs(normalY) > 1.0) {
		normalX /= normalY;
		if (normalY == INFINITY) {
			normalY = 1.0;
		}
		else {
			normalY /= normalY;
		}
	}*/

	return Vector2D(normalX, normalY);
}

// Function to calculate the reflection vector given the incident vector and the normal vector
__host__ __device__ Vector2D Math::calculateReflectionVector(const Vector2D& incidentVector, const Vector2D& normalVector) {
	// Calculate the dot product of the incident vector and the normal vector
	double dotProduct = incidentVector.X * normalVector.X + incidentVector.Y * normalVector.Y;

	// Calculate the reflection vector
	Vector2D reflectionVector = Vector2D(incidentVector.X - 2 * dotProduct * normalVector.X,
		incidentVector.Y - 2 * dotProduct * normalVector.Y);

	return reflectionVector;
}

// Function to compute the squared distance between two points
__host__ __device__ float Math::squared_distance(const Vector2D& p1, const Vector2D& p2) {
	float dx = p1.X - p2.X;
	float dy = p1.Y - p2.Y;
	return dx * dx + dy * dy;
}

// Check if the line segment AB intersects the circle with center C and radius R
__host__ __device__ bool Math::check_line_segment_circle_intersection(const Vector2D& A, const Vector2D& B, const Vector2D& C, float radius) {
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
	Vector2D P;
	P.X = A.X + (B.X - A.X) * dot_product / dist_AB_squared;
	P.Y = A.Y + (B.Y - A.Y) * dot_product / dist_AB_squared;

	// Check if the distance from P to C is less than or equal to the radius
	float dist_CP_squared = squared_distance(C, P);
	return dist_CP_squared <= radius * radius;
}

__host__ __device__ float Math::smoothingKernel(float radius, float distance) {
	if (distance >= radius) {
		return 0.0f;
	}

	float x = (radius - distance) / radius;
	return x * x;
}

__host__ __device__ float Math::smoothingKernelDerivative(float radius, float distance) {
	if (distance >= radius) {
		return 0.0f;
	}
	float x = (radius - distance) / radius;
	return 2 * x;
}

__host__ __device__ float Math::viscositySmoothingKernel(float radius, float distance) {
	if (distance >= radius) {
		return 0.0f;
	}
	float x = (radius * radius - distance * distance) / (radius * radius);
	return x * x * x;
}