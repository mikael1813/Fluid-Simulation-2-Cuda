#include "GUIApplication.hpp"
#include "Application.hpp"

#include "Observer.h"

class System : public Observer{
public:
	System() {};
	~System();
	void init();
	void startApp();

	void update(const std::string& app, const std::string& task, const float value) override;

	void updateGravity(float gravity) {
		m_App->updateGravity(gravity);
	}

	void updatePressure(float pressure) {
		m_App->updatePressure(pressure);
	}

	void updateTargetDensity(float targetDensity) {
		m_App->updateTargetDensity(targetDensity);
	}

	void updateViscosity(float viscosity) {
		m_App->updateViscosity(viscosity);
	}

	std::thread thread;

private:
	Application* m_App;
	GUIApplication* m_Gui;

	bool appCreated = false;
};