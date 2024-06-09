#include "System.hpp"
#include <thread>
#include "constants.h"


System::~System()
{
	delete m_Gui;
}

void startGui(GUIApplication* gui) {
	gui->init();
}

void System::init() {
	m_Gui = new GUIApplication();
	thread = std::thread(startGui, m_Gui);
	m_Gui->attach(this);
	this->startApp();
}

void System::startApp() {
	while (!m_Gui->scenarioStarted) {
		continue;
	}
	int selectedScenario = m_Gui->getSelectedScenario();

	m_App = new Application(selectedScenario);
	appCreated = true;
	m_App->attach(this);
	m_App->loop();
	delete m_App;
	appCreated = false;
	m_Gui->scenarioStarted = false;

	this->startApp();
}

void System::update(const std::string& app, const std::string& task, const float value) {
	if (app == GUI && appCreated) {
		if (task == UPDATE_GRAVITY) {
			m_App->updateGravity(value);
		}
		else if (task == UPDATE_PRESSURE) {
			m_App->updatePressure(value);
		}
		else if (task == UPDATE_TARGET_DENSITY) {
			m_App->updateTargetDensity(value);
		}
		else if (task == UPDATE_VISCOSITY) {
			m_App->updateViscosity(value);
		}
	}
	else if (app == APP) {
		
		if (task==TOP_WALL_PRESSURE) {
			m_Gui->updateTopWallPressure(value);
		}
		else if (task == BOTTOM_WALL_PRESSURE) {
			m_Gui->updateBottomWallPressure(value);
		}
		else if (task == LEFT_WALL_PRESSURE) {
			m_Gui->updateLeftWallPressure(value);
		}
		else if (task == RIGHT_WALL_PRESSURE) {
			m_Gui->updateRightWallPressure(value);
		}
		else if (task == FPS) {
			m_Gui->updateFPS(value);
		}
	}
}