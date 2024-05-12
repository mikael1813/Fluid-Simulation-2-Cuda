#pragma once

namespace constants {
	constexpr double m_PI = 3.14159265358979323846f;
}

#include <cmath> // Include <cmath> for handling infinity

struct Vector2D {

	Vector2D() {
		X = 0;
		Y = 0;
	}

	Vector2D(float x, float y) {
		X = x;
		Y = y;
	}

	// Function to calculate the magnitude of a 2D vector (x, y)
	float getMagnitude() const {
		return std::sqrt(X * X + Y * Y);
	}

	float getMagnitudeSquared() const {
		return X * X + Y * Y;
	}

	static Vector2D getRandomDirection();

	// add 2 vectors
	Vector2D operator+(const Vector2D& other) const {
		return Vector2D(X + other.X, Y + other.Y);
	}

	// multiply by a scalar
	Vector2D operator*(float scalar) const {
		return Vector2D(X * scalar, Y * scalar);
	}

	// dot product
	float operator*(const Vector2D& other) const {
		return X * other.X + Y * other.Y;
	}

	// add 1 vector to current vector
	Vector2D& operator+=(const Vector2D& other) {
		X += other.X;
		Y += other.Y;
		return *this;
	}

	// substract 2 vectors
	Vector2D operator-(const Vector2D& other) const {
		return Vector2D(X - other.X, Y - other.Y);
	}

	// subtract a vector from the current vector
	Vector2D& operator-=(const Vector2D& other) {
		X -= other.X;
		Y -= other.Y;
		return *this;
	}

	// divide by a scalar
	Vector2D operator/(float scalar) const {
		return Vector2D(X / scalar, Y / scalar);
	}

	// float times vector
	friend Vector2D operator*(float scalar, const Vector2D& vector) {
		return vector * scalar;
	}

	// minus vector
	Vector2D operator-() const {
		return Vector2D(-X, -Y);
	}

	float X;
	float Y;
};

struct Surface2D {

	Surface2D() {
		Point1 = Vector2D();
		Point2 = Vector2D();
	}

	Surface2D(Vector2D point1, Vector2D point2) {
		Point1 = point1;
		Point2 = point2;
	}

	Surface2D(float x1, float y1, float x2, float y2) {
		Point1 = Vector2D(x1, y1);
		Point2 = Vector2D(x2, y2);
	}

	Vector2D Point1;
	Vector2D Point2;
};



namespace Math {

	// Function to compute the squared distance between two points
	float squared_distance(const Vector2D& p1, const Vector2D& p2);

	// Function to calculate the slope of a line given two points
	double calculateSlope(Vector2D a, Vector2D b);

	// Function to calculate the normal vector given the slope of the surface line
	Vector2D calculateNormalVector(double surfaceLineSlope);

	// Function to calculate the reflection vector given the incident vector and the normal vector
	Vector2D calculateReflectionVector(const Vector2D& incidentVector, const Vector2D& normalVector);

	// Check if the line segment AB intersects the circle with center C and radius R
	bool check_line_segment_circle_intersection(const Vector2D& A, const Vector2D& B, const Vector2D& C, float radius);

	// Function to calculate the smoothing kernel
	float smoothingKernel(float radius, float distance);

	// Function to calculate the derivative of the smoothing kernel
	float smoothingKernelDerivative(float radius, float distance);

	// Function to calculate the viscosity smoothing kernel
	float viscositySmoothingKernel(float radius, float distance);

	// Function to calculate the first power of 2 bigger than n
	unsigned int nextPowerOf2(unsigned int n);

};