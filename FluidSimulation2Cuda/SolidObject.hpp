#pragma once

#include "Phisics.hpp"
#include <vector>
#include "Particle.hpp"

class SolidObject {
public:
	float m_Mass;
	float m_Volume;
	float m_Density;

	Vector2D m_Position;
	Vector2D m_PreviousPositon;
	Vector2D m_Velocity;
	Vector2D m_FutureVelocity;

	virtual void update(float dt) {
		m_Position = m_Position + m_Velocity * dt;
		m_Velocity = m_FutureVelocity;
	}

private:

};

class SolidRectangle : public SolidObject {
public:

	Surface2D leftSide;
	Surface2D rightSide;
	Surface2D topSide;
	Surface2D bottomSide;
	float m_Width = 0;
	float m_Height = 0;

	SolidRectangle() {

	}

	SolidRectangle(float width, float height, float density, Vector2D position) {

		m_Width = width;
		m_Height = height;

		m_Mass = width * height * density;
		m_Volume = width * height;
		m_Density = density;

		m_Position = position;
		m_Velocity = Vector2D();

		leftSide = Surface2D(Vector2D(position.X - width / 2, position.Y - height / 2), 
			Vector2D(position.X - width / 2, position.Y + height / 2));
		
		rightSide = Surface2D(Vector2D(position.X + width / 2, position.Y - height / 2),
			Vector2D(position.X + width / 2, position.Y + height / 2));

		topSide = Surface2D(Vector2D(position.X - width / 2, position.Y + height / 2),
			Vector2D(position.X + width / 2, position.Y + height / 2));

		bottomSide = Surface2D(Vector2D(position.X - width / 2, position.Y - height / 2),
			Vector2D(position.X + width / 2, position.Y - height / 2));
	}

	void update(float dt) override {

		m_Velocity = m_FutureVelocity;

		Vector2D gravity(0.0f, GRAVITY);

		m_Velocity += gravity * dt;

		m_PreviousPositon = m_Position;
		m_Position = m_Position + m_Velocity * dt;
		Vector2D positionChange = m_Position - m_PreviousPositon;

		leftSide.Point1 = leftSide.Point1 + positionChange;
		leftSide.Point2 = leftSide.Point2 + positionChange;

		rightSide.Point1 = rightSide.Point1 + positionChange;
		rightSide.Point2 = rightSide.Point2 + positionChange;

		topSide.Point1 = topSide.Point1 + positionChange;
		topSide.Point2 = topSide.Point2 + positionChange;

		bottomSide.Point1 = bottomSide.Point1 + positionChange;
		bottomSide.Point2 = bottomSide.Point2 + positionChange;

	}
};