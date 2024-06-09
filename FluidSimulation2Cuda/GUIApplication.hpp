#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <vector>
#include <string>

#include "Subject.h"

class System;

class GUIApplication : public Subject {
public:
	GUIApplication();
	~GUIApplication();
	void init();
	void run();

	int getSelectedScenario();
	bool scenarioStarted = false;

	void updateTopWallPressure(float pressure);
	void updateBottomWallPressure(float pressure);
	void updateLeftWallPressure(float pressure);
	void updateRightWallPressure(float pressure);

	void updateFPS(float fps);

private:
	SDL_Window* window;
	SDL_Renderer* renderer;
	TTF_Font* font;
	bool quit;
	SDL_Rect startButton;
	SDL_Rect editButton;
	std::vector<std::string> scenarioOptions;
	int selectedScenario;
	bool dropdownVisible;
	bool firstTime = true;

	float topWallPressure;
	float bottomWallPressure;
	float leftWallPressure;
	float rightWallPressure;

	float newTopWallPressure;
	float newBottomWallPressure;
	float newLeftWallPressure;
	float newRightWallPressure;

	float fps;

	struct Slider {
		SDL_Rect bar;
		SDL_Rect handle;
		float minValue;
		float maxValue;
		float value;
		std::string label;
	};

	Slider gravitySlider;
	Slider pressureSlider;
	Slider targetDensitySlider;
	Slider viscositySlider;

	void start();
	void renderButton(int x, int y, int w, int h, const char* text);
	bool isInside(int x, int y, int rectX, int rectY, int rectW, int rectH);
	void renderDropdown(int x, int y, int w, int h, const std::vector<std::string>& options, int selectedIndex);
	int handleDropdownClick(int mouseX, int mouseY, int x, int y, int w, int h, const std::vector<std::string>& options);
	void renderSlider(Slider& slider);
	void handleSlider(Slider& slider, int mouseX, int mouseY);
	void handleEvents();
	void render();
	void renderFPS();
	void renderPressureSquare();
	float GUIApplication::getActualSliderValue(Slider& slider);
};