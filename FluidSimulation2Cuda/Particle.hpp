#pragma once

#include "Phisics.hpp"

#include <iostream>
#include <vector>

constexpr float GRAVITY = 10.0f;


class Particle {
public:
	Particle(){}
	Particle(float x, float y, int id) : m_Position(Vector2D(x, y)), m_ID(id), m_TemporaryVelocity(Vector2D()) {}
	Vector2D m_PredictedPosition;
	Vector2D m_LastSafePosition;
	Vector2D m_FutureVelocity;

	Vector2D m_TemporaryVelocity;


	float m_Density = 0.0f;
	int m_ID;

	void update(float dt) {
		if (dt == 0) {
			return;
		}

		m_LastSafePosition = m_Position;

		Vector2D gravity(0.0f, GRAVITY);

		m_Velocity += gravity * dt;

		/*for (auto& force : m_Forces) {
			m_Velocity += force * dt;
		}

		m_Forces.clear();*/

		m_Velocity += m_TemporaryVelocity * dt;

		m_TemporaryVelocity = Vector2D();

		m_Position += m_Velocity * dt;

		//m_Velocity = m_Velocity * 0.95f;
	}

	void updateVelocity() {
		m_Velocity = m_FutureVelocity;
	}

	Vector2D getVelocity() {
		return m_Velocity;
	}

	void setVelocity(Vector2D velocity) {
		m_Velocity = velocity;
	}

	Vector2D getPosition() {
		return m_Position;
	}

	void setPosition(Vector2D position) {
		m_Position = position;
	}

	void addForce(Vector2D force) {
		m_TemporaryVelocity += force / mass;
	}

private:

	Vector2D m_Position;
	Vector2D m_Velocity;

	//std::vector<Vector2D> m_Forces;

	float visible_radius = 2.0f;
	float mass = 1.0f;
};