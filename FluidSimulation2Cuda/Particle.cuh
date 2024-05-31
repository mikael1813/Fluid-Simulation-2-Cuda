#pragma once

#include "Phisics.cuh"

#include <iostream>
#include <vector>

constexpr float GRAVITY = 10.0f;


class Particle {
public:
	__host__ __device__ Particle(){}
	__host__ __device__ Particle(float x, float y, int id) : m_Position(Vector2D(x, y)), m_ID(id), m_TemporaryVelocity(Vector2D()) {}
	Vector2D m_PredictedPosition;
	Vector2D m_LastSafePosition;
	Vector2D m_FutureVelocity;

	Vector2D m_TemporaryVelocity;

	Vector2D m_Velocity;
	Vector2D m_Position;

	float m_Density = 0.0f;
	int m_ID;

	__host__ __device__ void update(float dt) {
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

	__host__ __device__ void updateVelocity() {
		m_Velocity = m_FutureVelocity;
	}

	__host__ __device__ Vector2D getVelocity() {
		return m_Velocity;
	}

	__host__ __device__ void setVelocity(Vector2D velocity) {
		m_Velocity = velocity;
	}

	__host__ __device__ Vector2D getPosition() {
		return m_Position;
	}

	__host__ __device__ void setPosition(Vector2D position) {
		m_Position = position;
	}

	__host__ __device__ void addForce(Vector2D force) {
		m_TemporaryVelocity += force / mass;
	}

private:

	

	//std::vector<Vector2D> m_Forces;

	float visible_radius = 2.0f;
	float mass = 1.0f;
};