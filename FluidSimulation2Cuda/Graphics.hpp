#pragma once

#include <GLFW/glfw3.h>
#include <corecrt_math.h>

#include "Phisics.hpp"
#include "SolidObject.hpp"
#include <utility>

namespace Graphics {

	void DrawCircle(int width, int height, float x, float y, float radius, int num_segments) {

		float raport = (float)width / (float)height;

		x = (2.0f * x) / width - 1.0f;
		x = x * raport;

		y = 1.0f - (2.0f * y) / height;

		radius = (2.0f * radius) / width;

		glBegin(GL_TRIANGLE_FAN);
		for (int ii = 0; ii < num_segments; ii++) {
			float theta = 2.0f * 3.1415 * float(ii) / float(num_segments);
			float dx = radius * cos(theta);
			float dy = radius * sin(theta);
			glVertex2f(x + dx, y + dy);
		}
		glEnd();
	}

	void DrawLine(int width, int height, Vector2D a, Vector2D b) {

		float raport = (float)width / (float)height;

		a.X = (2.0f * a.X) / width - 1.0f;
		a.X = a.X * raport;

		a.Y = 1.0f - (2.0f * a.Y) / height;

		b.X = (2.0f * b.X) / width - 1.0f;
		b.X = b.X * raport;

		b.Y = 1.0f - (2.0f * b.Y) / height;

		glBegin(GL_LINES);
		glVertex2f(a.X, a.Y);
		glVertex2f(b.X, b.Y);
		glEnd();
	}

	void DrawRectangle(int width, int height, SolidRectangle rectangle) {

		DrawLine(width, height, rectangle.leftSide.Point1, rectangle.leftSide.Point2);
		DrawLine(width, height, rectangle.rightSide.Point1, rectangle.rightSide.Point2);
		DrawLine(width, height, rectangle.topSide.Point1, rectangle.topSide.Point2);
		DrawLine(width, height, rectangle.bottomSide.Point1, rectangle.bottomSide.Point2);
	}

	//void drawSurface2D(SDL_Renderer* renderer, Surface2D surface) {
	//	SDL_RenderDrawLine(renderer, surface.Point1.X, surface.Point1.Y, surface.Point2.X, surface.Point2.Y);
	//}

	// Function to map velocity to color (blue for velocity 0, red for max velocity)
	void velocityToColor(float velocity, float& red, float& green, float& blue) {
		float maxVelocity = 50.0f; // Maximum velocity in the simulation

		// Normalize velocity between 0 and 1
		float normalizedVelocity = std::min(std::max(velocity / maxVelocity, 0.0f), 1.0f);

		// Linear interpolation between blue and red based on normalized velocity
		red = normalizedVelocity;      // Red component increases with velocity
		green = 0.5f - normalizedVelocity / 2;                  // No green component
		blue = 1.0f - normalizedVelocity; // Blue component decreases with velocity
	}

} // namespace Graphics