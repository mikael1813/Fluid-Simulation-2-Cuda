#include "InteractionMatrixClass.hpp"

InteractionMatrixClass* InteractionMatrixClass::s_Instance = nullptr;

InteractionMatrixClass::~InteractionMatrixClass() {
	delete s_Instance;
}

void InteractionMatrixClass::initializeMatrix(int width, int height, int particleRadiusOfRepel)
{
	for (int i = 0; i <= height / particleRadiusOfRepel; i++) {
		std::vector<MatrixComponenets> row;
		for (int j = 0; j <= width / particleRadiusOfRepel; j++) {
			row.push_back(MatrixComponenets());
		}
		m_InteractionsMatrix.push_back(row);
	}
}

void InteractionMatrixClass::addToInteractionMatrixCellSurroundingCells(int x, int y, std::vector<std::vector<MatrixComponenets>>& temporary) {

	if (x < 0 || x >= m_InteractionsMatrix.size() || y < 0 || y >= m_InteractionsMatrix.at(0).size()) {
		return;
	}

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if (x + i < 0 || x + i >= m_InteractionsMatrix.size() || y + j < 0 || y + j >= m_InteractionsMatrix.at(0).size()) {
				continue;
			}
			m_InteractionsMatrix.at(x).at(y).particles.insert(m_InteractionsMatrix.at(x).at(y).particles.end(),
				temporary.at(x + i).at(y + j).particles.begin(),
				temporary.at(x + i).at(y + j).particles.end());
		}
	}
}

void InteractionMatrixClass::updateInteractionMatrix(const std::vector<Particle*>& particles, int particleRadiusOfRepel)
{
	// Parallelize the loop using OpenMP
	//#pragma omp parallel for
	std::vector<std::vector<MatrixComponenets>> temporary;


	for (int i = 0; i < m_InteractionsMatrix.size(); i++) {
		std::vector<MatrixComponenets> row;
		for (int j = 0; j < m_InteractionsMatrix.at(0).size(); j++) {
			m_InteractionsMatrix.at(i).at(j).particles.clear();
			row.push_back(MatrixComponenets());
		}
		temporary.push_back(row);
	}

	for (int i = 0; i < particles.size(); i++) {
		int x = particles.at(i)->m_PredictedPosition.Y / particleRadiusOfRepel;
		int y = particles.at(i)->m_PredictedPosition.X / particleRadiusOfRepel;
		if (x < 0 || x >= m_InteractionsMatrix.size() || y < 0 || y >= m_InteractionsMatrix.at(0).size()) {
			continue;
		}

		temporary.at(x).at(y).particles.push_back(particles.at(i));

	}

	if (particles.size() < m_InteractionsMatrix.size() * m_InteractionsMatrix.at(0).size()) {

		for (auto& particle : particles) {
			int x = particle->m_PredictedPosition.Y / particleRadiusOfRepel;
			int y = particle->m_PredictedPosition.X / particleRadiusOfRepel;

			if (x < 0 || x >= m_InteractionsMatrix.size() || y < 0 || y >= m_InteractionsMatrix.at(0).size() || m_InteractionsMatrix.at(x).at(y).particles.size() > 0) {
				continue;
			}

			this->addToInteractionMatrixCellSurroundingCells(x, y, temporary);
		}

		return;
	}

	for (int x = 0; x < m_InteractionsMatrix.size(); x++) {
		for (int y = 0; y < m_InteractionsMatrix.at(0).size(); y++) {

			this->addToInteractionMatrixCellSurroundingCells(x, y, temporary);
		}
	}
}

std::vector<Particle*> InteractionMatrixClass::getParticlesInCell(Vector2D particlePosition, int particleRadiusOfRepel) {
	std::vector<Particle*> output;

	int x = particlePosition.Y / particleRadiusOfRepel;
	int y = particlePosition.X / particleRadiusOfRepel;

	if (x < 0 || x >= m_InteractionsMatrix.size() || y < 0 || y >= m_InteractionsMatrix.at(0).size()) {
		return output;
	}

	return m_InteractionsMatrix.at(x).at(y).particles;
}