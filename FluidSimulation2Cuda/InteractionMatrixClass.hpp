#pragma once

#include <vector>

#include "Particle.hpp"

struct MatrixComponenets {
	std::vector<Particle*> particles;
	std::vector<Surface2D*> obstacles;
};

class InteractionMatrixClass {
public:
	void initializeMatrix(int width, int height, int particleRadiusOfRepel);

	void addToInteractionMatrixCellSurroundingCells(int x, int y, std::vector<std::vector<MatrixComponenets>>& temporary);
	void updateInteractionMatrix(const std::vector<Particle*>& particles, int particleRadiusOfRepel);
	std::vector<Particle*> getParticlesInCell(Vector2D particlePosition, int particleRadiusOfRepel);
	std::vector<std::vector<MatrixComponenets>>& getMatrix() { return m_InteractionsMatrix; }

	static InteractionMatrixClass* getInstance() {
		if (s_Instance == nullptr) {
			s_Instance = new InteractionMatrixClass();
		}
		return s_Instance;
	}

	InteractionMatrixClass(InteractionMatrixClass const&) = delete;
	void operator=(InteractionMatrixClass const&) = delete;

	/*void addParticle(Particle* particle);
	void addObstacle(Surface2D* obstacle);

	void clearMatrix();*/
	~InteractionMatrixClass();
private:
	InteractionMatrixClass() {}

	static InteractionMatrixClass* s_Instance;

	std::vector<std::vector<MatrixComponenets>> m_InteractionsMatrix;
};