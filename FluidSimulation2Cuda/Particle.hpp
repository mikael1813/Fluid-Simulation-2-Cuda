#pragma once

#include "Phisics.hpp"

#include <iostream>
#include <vector>

constexpr float GRAVITY = 10.0f;


class Particle {
public:
	Particle(){
		m_Exists = false;
	}
	Particle(float x, float y, int id) : m_Position(Vector2D(x, y)), m_ID(id), m_TemporaryVelocity(Vector2D()) {
		m_LastSafePosition = m_Position;
		m_Exists = true;
	}
	Vector2D m_PredictedPosition;
	Vector2D m_LastSafePosition;
	Vector2D m_FutureVelocity;

	Vector2D m_TemporaryVelocity;

	Vector2D m_Velocity;
	Vector2D m_Position;

	float m_Density = 0.0f;
	int m_ID;
	bool m_Exists;
	float m_Mass = 1.0f;

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

		m_Velocity += m_TemporaryVelocity;

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
		m_TemporaryVelocity += force / m_Mass;
	}

private:

	

	//std::vector<Vector2D> m_Forces;

	float visible_radius = 2.0f;
};