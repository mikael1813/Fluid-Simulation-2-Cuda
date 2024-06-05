#include "Phisics.hpp"

constexpr float targetDensity = 0.8f;

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

	__device__ GpuVector2D isPointFartherThanSideFromCorner(Vector2D point, GpuVector2D bottomLeftCorner, float sideLength) {
		// Coordinates of the corners of the square
		GpuVector2D bottomRightCorner = { bottomLeftCorner.X + sideLength, bottomLeftCorner.Y };
		GpuVector2D topLeftCorner = { bottomLeftCorner.X, bottomLeftCorner.Y + sideLength };
		GpuVector2D topRightCorner = { bottomLeftCorner.X + sideLength, bottomLeftCorner.Y + sideLength };

		// Calculate the distances from the point to each corner
		float distanceToBottomLeft = sqrt(pow(point.X - bottomLeftCorner.X, 2) + pow(point.Y - bottomLeftCorner.Y, 2));
		float distanceToBottomRight = sqrt(pow(point.X - bottomRightCorner.X, 2) + pow(point.Y - bottomRightCorner.Y, 2));
		float distanceToTopLeft = sqrt(pow(point.X - topLeftCorner.X, 2) + pow(point.Y - topLeftCorner.Y, 2));
		float distanceToTopRight = sqrt(pow(point.X - topRightCorner.X, 2) + pow(point.Y - topRightCorner.Y, 2));

		if (distanceToBottomLeft > sideLength) {
			return GpuVector2D(1, -1);
		}
		else if (distanceToBottomRight > sideLength) {
			return GpuVector2D(1, 1);
		}
		else if (distanceToTopLeft > sideLength) {
			return GpuVector2D(-1, -1);
		}
		else if (distanceToTopRight > sideLength) {
			return GpuVector2D(-1, 1);
		}

		return GpuVector2D(100, 100);
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

	__device__ float viscositySmoothingKernel(float radius, float distance) {
		if (distance >= radius) {
			return 0.0f;
		}
		float x = (radius * radius - distance * distance) / (radius * radius);
		return x * x * x;
	}

	__device__ float surfaceTensionSmoothingKernel(float particleDensity, float averageDensity) {
		if (particleDensity >= averageDensity) {
			return 0.0f;
		}
		float x = abs(averageDensity - particleDensity);
		return x * x;
	}

	__device__ float convertDensityToPressure(float density) {
		//targetDensity = 0.8f;
		//const float pressureConstant = 10.0f;
		const float pressureConstant = 50.0f;

		//float densityError = density <= targetDensity ? 0.0f : density - targetDensity;
		float densityError = density - targetDensity;

		float pressure = pressureConstant * densityError;
		return pressure;
	}

	__device__ float convertDensitiesToArhimedeInfluence(float objectDensity, float particlesSurroundingDensity) {
		return -particlesSurroundingDensity / objectDensity;
	}

	__device__ float calculateSharedPressure(float density1, float density2) {
		float pressure1 = convertDensityToPressure(density1);
		float pressure2 = convertDensityToPressure(density2);
		return (pressure1 + pressure2) / 2;
	}

	__device__ float min(float a, float b) {
		return a < b ? a : b;
	}

	__device__ float max(float a, float b) {
		return a > b ? a : b;
	}

	// Function to find orientation of triplet (p, q, r)
	__device__ int orientation(Vector2D p, Vector2D q, Vector2D r) {
		int val = (q.Y - p.Y) * (r.X - q.X) - (q.X - p.X) * (r.Y - q.Y);
		if (val == 0) return 0; // colinear
		return (val > 0) ? 1 : 2; // clock or counterclock wise
	}

	// Function to check if the segment 'p1q1' and 'p2q2' intersect
	__device__ bool doIntersect(Vector2D p1, Vector2D q1, Vector2D p2, Vector2D q2) {
		// Find the four orientations needed for general and special cases
		int o1 = orientation(p1, q1, p2);
		int o2 = orientation(p1, q1, q2);
		int o3 = orientation(p2, q2, p1);
		int o4 = orientation(p2, q2, q1);

		// General case
		if (o1 != o2 && o3 != o4)
			return true;

		// Special cases
		if (o1 == 0 && (p2.X >= min(p1.X, q1.X) && p2.X <= max(p1.X, q1.X)) && (p2.Y >= min(p1.Y, q1.Y) && p2.Y <= max(p1.Y, q1.Y)))
			return true;

		if (o2 == 0 && (q2.X >= min(p1.X, q1.X) && q2.X <= max(p1.X, q1.X)) && (q2.Y >= min(p1.Y, q1.Y) && q2.Y <= max(p1.Y, q1.Y)))
			return true;

		if (o3 == 0 && (p1.X >= min(p2.X, q2.X) && p1.X <= max(p2.X, q2.X)) && (p1.Y >= min(p2.Y, q2.Y) && p1.Y <= max(p2.Y, q2.Y)))
			return true;

		if (o4 == 0 && (q1.X >= min(p2.X, q2.X) && q1.X <= max(p2.X, q2.X)) && (q1.Y >= min(p2.Y, q2.Y) && q1.Y <= max(p2.Y, q2.Y)))
			return true;

		return false; // Doesn't fall in any of the above cases
	}

	__device__ GpuVector2D getCollisionPoint(Particle particle, SolidRectangle rectangle) {
		float halfWidth = (rectangle.rightSide.Point1.X - rectangle.leftSide.Point1.X) / 2.0f;
		float halfHeight = (rectangle.topSide.Point1.Y - rectangle.bottomSide.Point1.Y) / 2.0f;

		Vector2D rectangleCenter = rectangle.m_Position;
		Vector2D prevRectangleCenter = rectangle.m_PreviousPositon;

		// Check for collision along each edge with line intersection approach
		if (particle.m_Position.X < particle.m_LastSafePosition.X &&
			particle.m_Position.X >= rectangleCenter.X - halfWidth) {
			// Point moved left into the rectangle - collision with right edge
			float t = (rectangleCenter.X - halfWidth - prevRectangleCenter.X) /
				(particle.m_LastSafePosition.X - particle.m_Position.X);
			return { rectangleCenter.X - halfWidth,
					particle.m_LastSafePosition.Y + t * (particle.m_Position.Y - particle.m_LastSafePosition.Y) };
		}
		else if (particle.m_Position.X > particle.m_LastSafePosition.X &&
			particle.m_Position.X <= rectangleCenter.X + halfWidth) {
			// Point moved right into the rectangle - collision with left edge
			float t = (rectangleCenter.X + halfWidth - prevRectangleCenter.X) /
				(particle.m_LastSafePosition.X - particle.m_Position.X);
			return { rectangleCenter.X + halfWidth,
					particle.m_LastSafePosition.Y + t * (particle.m_Position.Y - particle.m_LastSafePosition.Y) };
		}
		else if (particle.m_Position.Y < particle.m_LastSafePosition.Y &&
			particle.m_Position.Y >= rectangleCenter.Y - halfHeight) {
			// Point moved up into the rectangle - collision with bottom edge
			float t = (rectangleCenter.Y - halfHeight - prevRectangleCenter.Y) /
				(particle.m_LastSafePosition.Y - particle.m_Position.Y);
			return { particle.m_LastSafePosition.X + t * (particle.m_Position.X - particle.m_LastSafePosition.X),
					rectangleCenter.Y - halfHeight };
		}
		else if (particle.m_Position.Y > particle.m_LastSafePosition.Y &&
			particle.m_Position.Y <= rectangleCenter.Y + halfHeight) {
			// Point moved down into the rectangle - collision with top edge
			float t = (rectangleCenter.Y + halfHeight - prevRectangleCenter.Y) /
				(particle.m_LastSafePosition.Y - particle.m_Position.Y);
			return { particle.m_LastSafePosition.X + t * (particle.m_Position.X - particle.m_LastSafePosition.X),
					rectangleCenter.Y + halfHeight };
		}
		else {
			// No collision detected or point was already inside
			return { -1.0f, -1.0f }; // Indicate no collision point
		}
	}



	__device__ GpuVector2D getExpulsionPoint(Particle particle, SolidRectangle rectangle) {
		float halfWidth = (rectangle.rightSide.Point1.X - rectangle.leftSide.Point1.X) / 2.0f;
		float halfHeight = (rectangle.topSide.Point1.Y - rectangle.bottomSide.Point1.Y) / 2.0f;

		Vector2D rectangleCenter = rectangle.m_Position;

		// Check which edge the point is closest to
		float distanceToLeft = particle.m_Position.X - rectangleCenter.X + halfWidth;
		float distanceToRight = rectangleCenter.X + halfWidth - particle.m_Position.X;
		float distanceToTop = rectangleCenter.Y + halfHeight - particle.m_Position.Y;
		float distanceToBottom = particle.m_Position.Y - rectangleCenter.Y + halfHeight;

		float minDistance = min(min(distanceToLeft, distanceToRight), min(distanceToTop, distanceToBottom));

		if (minDistance == distanceToLeft) {
			// Point closest to left edge - expel to the right
			return { particle.m_Position.X + 2 * distanceToLeft, particle.m_Position.Y };
		}
		else if (minDistance == distanceToRight) {
			// Point closest to right edge - expel to the left
			return { particle.m_Position.X - 2 * distanceToRight, particle.m_Position.Y };
		}
		else if (minDistance == distanceToTop) {
			// Point closest to top edge - expel to the bottom
			return { particle.m_Position.X, particle.m_Position.Y - 2 * distanceToTop };
		}
		else {
			// Point closest to bottom edge - expel to the top
			return { particle.m_Position.X, particle.m_Position.Y + 2 * distanceToBottom };
		}
	}

} // namespace CudaMath