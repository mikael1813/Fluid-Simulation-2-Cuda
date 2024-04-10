#include "Pipe.hpp"
#include <random>
#include <list>
#include "Phisics.hpp"

//void Pipe::update(float dt, std::vector<Particle*>& particles, std::vector<Particle*> surroundingParticles, float particleSize) {


//}

void GeneratorPipe::update(float dt, std::vector<Particle*>& particles, std::vector<Particle*> surroundingParticles, float particleSize)
{
	std::vector<Particle*> particlesToAdd;

	// Seed the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	//for (int i = 0; i < constants::m_PI * m_InteractionRadius * m_InteractionRadius / (particleSize * 5000); i++) {
	while (particlesToAdd.size() < m_ParticlesPerCycle) {
		int posX = std::uniform_int_distribution<int>(m_Position.X - m_InteractionRadius / 2, m_Position.X + m_InteractionRadius / 2)(gen);
		int posY = std::uniform_int_distribution<int>(m_Position.Y - m_InteractionRadius / 2, m_Position.Y + m_InteractionRadius / 2)(gen);

		particlesToAdd.push_back(new Particle(posX, posY, m_ID++));
	}

	// Mark elements for deletion
	std::vector<bool> marked_for_deletion(particlesToAdd.size(), false);

	for (auto& particle : surroundingParticles) {
		Vector2D direction = particle->getPosition() - m_Position;

		if (direction.getMagnitude() == 0) {
			direction = Vector2D::getRandomDirection();
		}

		float distance = direction.getMagnitude();
		if (distance <= m_InteractionRadius) {
			particle->addForce(direction / distance * m_Pressure);
		}

		// here lies a error down below
		// to do: solve it

		for (int i = 0; i < particlesToAdd.size(); i++) {

			Particle* otherParticle = particlesToAdd.at(i);

			float distance2 = (particle->getPosition() - otherParticle->getPosition()).getMagnitude();

			if (distance2 <= particleSize) {
				marked_for_deletion.at(i) = true;
			}
		}

		// Remove marked elements from the vector
		for (int i = 0; i < particlesToAdd.size(); i++) {
			if (marked_for_deletion.at(i)) {
				delete particlesToAdd.at(i);
				particlesToAdd.erase(particlesToAdd.begin() + i);
				marked_for_deletion.erase(marked_for_deletion.begin() + i);
				i--;
			}
		}
	}

	for (auto& particle : particlesToAdd) {
		particles.push_back(particle);
	}
}

void ConsumerPipe::update(float dt, std::vector<Particle*>& particles, std::vector<Particle*> surroundingParticles, float particleSize)
{
	std::vector<Particle*> particlesToRemove;

	for (auto& particle : particles) {
		Vector2D direction = particle->getPosition() - m_Position;

		if (direction.getMagnitude() == 0) {
			direction = Vector2D::getRandomDirection();
		}

		float distance = direction.getMagnitude();
		if (distance <= m_InteractionRadius + particleSize) {
			particlesToRemove.push_back(particle);
		}

		if (particlesToRemove.size() >= m_ParticlesPerCycle) {
			break;
		}
	}

	std::vector<int> particlesToRemoveIndex;

	for (auto& particle : particlesToRemove) {
		for (int i = 0; i < particles.size(); i++) {
			if (particle == particles.at(i)) {
				particlesToRemoveIndex.push_back(i);
				break;
			}
		}
	}

	std::sort(particlesToRemoveIndex.begin(), particlesToRemoveIndex.end());

	int particlesRemoved = 0;

	for (auto& index : particlesToRemoveIndex) {
		delete particles.at(index - particlesRemoved);
		particles.erase(particles.begin() + index - particlesRemoved);
		particlesRemoved++;
	}

}
