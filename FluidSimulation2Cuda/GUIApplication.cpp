#include "GUIApplication.hpp"

#include <vector>
#include <thread>
#include <sstream>
#include <iomanip>

#include "constants.h"

GUIApplication::GUIApplication()
	: window(nullptr), renderer(nullptr), font(nullptr), quit(false),
	startButton({ 300, 200, 200, 50 }), editButton({ 200, 300, 400, 50 }),
	selectedScenario(-1), dropdownVisible(false),
	gravitySlider({ {300, 150, 200, 10}, {300, 145, 20, 20}, 0.0f, 100.0f, 0.0f, "Gravity" }),
	pressureSlider({ {300, 200, 200, 10}, {300, 195, 20, 20}, 0.0f, 100.0f, 0.0f, "Pressure" }),
	targetDensitySlider({ {300, 250, 200, 10}, {300, 245, 20, 20}, 0.0f, 2.0f, 0.0f, "Target Density" }),
	viscositySlider({ {300, 300, 200, 10}, {300, 295, 20, 20}, 0.0f, 1.0f, 0.1f, "Viscosity" }),
	topWallPressure(0.0f), bottomWallPressure(0.0f), leftWallPressure(0.0f), rightWallPressure(0.0f),
	newTopWallPressure(0.0f), newBottomWallPressure(0.0f), newLeftWallPressure(0.0f), newRightWallPressure(0.0f) {}

GUIApplication::~GUIApplication() {
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	TTF_CloseFont(font);
	TTF_Quit();
	SDL_Quit();
}

void GUIApplication::init() {

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
		exit(1);
	}

	if (TTF_Init() == -1) {
		std::cerr << "SDL_ttf could not initialize! TTF_Error: " << TTF_GetError() << std::endl;
		SDL_Quit();
		exit(1);
	}

	font = TTF_OpenFont("Arial.ttf", 24);
	if (!font) {
		std::cerr << "Failed to load font: " << TTF_GetError() << std::endl;
		TTF_Quit();
		SDL_Quit();
		exit(1);
	}

	window = SDL_CreateWindow("Simple GUI", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 600, SDL_WINDOW_SHOWN);
	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		TTF_CloseFont(font);
		TTF_Quit();
		SDL_Quit();
		exit(1);
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (!renderer) {
		std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(window);
		TTF_CloseFont(font);
		TTF_Quit();
		SDL_Quit();
		exit(1);
	}

	scenarioOptions = { "Scenario 1: Normal", "Scenario  2: Obstacle", "Scenario 3: Obstacle + Sink","Scenario 4: WaterFall", "Scenario  5: Dam", "Scenario 6: Dam" };

	this->run();
}

void GUIApplication::run() {
	while (!quit) {
		handleEvents();
		render();
	}
}

int GUIApplication::getSelectedScenario()
{
	return selectedScenario;
}

void GUIApplication::updateTopWallPressure(float pressure)
{
	newTopWallPressure += pressure;
}

void GUIApplication::updateBottomWallPressure(float pressure)
{
	newBottomWallPressure += pressure;
}

void GUIApplication::updateLeftWallPressure(float pressure)
{
	newLeftWallPressure += pressure;
}

void GUIApplication::updateRightWallPressure(float pressure)
{
	newRightWallPressure += pressure;
}

void GUIApplication::updateFPS(float fps)
{
	this->fps = fps;

	int transformingUnit = 1000;

	topWallPressure = abs(newTopWallPressure) / transformingUnit;
	bottomWallPressure = newBottomWallPressure / transformingUnit;
	leftWallPressure = abs(newLeftWallPressure) / transformingUnit;
	rightWallPressure = newRightWallPressure / transformingUnit;

	newTopWallPressure = 0.0f;
	newBottomWallPressure = 0.0f;
	newLeftWallPressure = 0.0f;
	newRightWallPressure = 0.0f;
}

void GUIApplication::renderButton(int x, int y, int w, int h, const char* text) {
	SDL_Rect buttonRect = { x, y, w, h };
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderFillRect(renderer, &buttonRect);

	SDL_Color textColor = { 0, 0, 0, 255 };
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, text, textColor);
	SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	int textW = 0, textH = 0;
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	SDL_Rect textRect = { x + (w - textW) / 2, y + (h - textH) / 2, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);
}

bool GUIApplication::isInside(int x, int y, int rectX, int rectY, int rectW, int rectH) {
	return x > rectX && x < rectX + rectW && y > rectY && y < rectY + rectH;
}

void GUIApplication::renderDropdown(int x, int y, int w, int h, const std::vector<std::string>& options, int selectedIndex) {
	for (int i = 0; i < options.size(); ++i) {
		SDL_Rect optionRect = { x, y + i * h, w, h };

		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderFillRect(renderer, &optionRect);

		SDL_Color textColor = { 0, 0, 0, 255 };
		SDL_Surface* textSurface = TTF_RenderText_Solid(font, options[i].c_str(), textColor);
		SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

		int textW = 0, textH = 0;
		SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
		SDL_Rect textRect = { x + (w - textW) / 2, y + i * h + (h - textH) / 2, textW, textH };

		SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

		SDL_DestroyTexture(textTexture);
		SDL_FreeSurface(textSurface);
	}
}

int GUIApplication::handleDropdownClick(int mouseX, int mouseY, int x, int y, int w, int h, const std::vector<std::string>& options) {
	for (int i = 0; i < options.size(); ++i) {
		if (isInside(mouseX, mouseY, x, y + i * h, w, h)) {
			return i;
		}
	}
	return -1;
}

float GUIApplication::getActualSliderValue(Slider& slider) {
	return (slider.minValue + slider.maxValue) * slider.value;
}

void GUIApplication::renderSlider(Slider& slider) {
	// Render the bar
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderFillRect(renderer, &slider.bar);

	// Render the handle
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderFillRect(renderer, &slider.handle);

	// Render the label
	SDL_Color textColor = { 255, 255, 255, 255 };
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, slider.label.c_str(), textColor);
	SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	int textW = 0, textH = 0;
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	SDL_Rect textRect = { slider.bar.x - textW - 10, slider.bar.y - (textH / 2) + 5, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	// Render the min, max, and current values
	std::stringstream ss;
	ss << slider.minValue;
	std::string minValueText = ss.str();
	ss.str("");

	ss << slider.maxValue;
	std::string maxValueText = ss.str();
	ss.str("");

	ss << std::fixed << std::setprecision(2) << getActualSliderValue(slider);
	std::string valueText = ss.str();

	textSurface = TTF_RenderText_Solid(font, minValueText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { slider.bar.x - textW / 2, slider.bar.y + slider.bar.h + 5, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	textSurface = TTF_RenderText_Solid(font, maxValueText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { slider.bar.x + slider.bar.w - textW / 2, slider.bar.y + slider.bar.h + 5, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	textSurface = TTF_RenderText_Solid(font, valueText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { slider.bar.x + (slider.bar.w - textW) / 2, slider.bar.y + slider.bar.h + 5, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);
}

void GUIApplication::renderFPS() {
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << fps << " FPS";
	std::string fpsText = ss.str();

	SDL_Color textColor = { 255, 255, 255, 255 };
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, fpsText.c_str(), textColor);
	SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	int textW = 0, textH = 0;
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	SDL_Rect textRect = { 10, 10, textW, textH };

	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);
}

void GUIApplication::renderPressureSquare() {
	SDL_Rect square = { 300, 400, 150, 150 };

	// Draw the square
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderDrawRect(renderer, &square);

	// Draw the pressure values
	std::stringstream ss;

	// Top wall pressure
	ss << std::fixed << std::setprecision(2) << topWallPressure;
	std::string topText = ss.str();
	ss.str("");
	SDL_Color textColor = { 255, 255, 255, 255 };
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, topText.c_str(), textColor);
	SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
	int textW = 0, textH = 0;
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	SDL_Rect textRect = { square.x + (square.w - textW) / 2, square.y - textH - 5, textW, textH };
	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	// Bottom wall pressure
	ss << std::fixed << std::setprecision(2) << bottomWallPressure;
	std::string bottomText = ss.str();
	ss.str("");
	textSurface = TTF_RenderText_Solid(font, bottomText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { square.x + (square.w - textW) / 2, square.y + square.h + 5, textW, textH };
	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	// Left wall pressure
	ss << std::fixed << std::setprecision(2) << leftWallPressure;
	std::string leftText = ss.str();
	ss.str("");
	textSurface = TTF_RenderText_Solid(font, leftText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { square.x - textW - 5, square.y + (square.h - textH) / 2, textW, textH };
	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);

	// Right wall pressure
	ss << std::fixed << std::setprecision(2) << rightWallPressure;
	std::string rightText = ss.str();
	ss.str("");
	textSurface = TTF_RenderText_Solid(font, rightText.c_str(), textColor);
	textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
	SDL_QueryTexture(textTexture, NULL, NULL, &textW, &textH);
	textRect = { square.x + square.w + 5, square.y + (square.h - textH) / 2, textW, textH };
	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
	SDL_DestroyTexture(textTexture);
	SDL_FreeSurface(textSurface);
}

void GUIApplication::handleSlider(Slider& slider, int mouseX, int mouseY) {
	if (scenarioStarted && isInside(mouseX, mouseY, slider.bar.x, slider.bar.y - 5, slider.bar.w, slider.bar.h + 10)) {
		slider.handle.x = mouseX - slider.handle.w / 2;
		if (slider.handle.x < slider.bar.x) slider.handle.x = slider.bar.x;
		if (slider.handle.x > slider.bar.x + slider.bar.w - slider.handle.w) slider.handle.x = slider.bar.x + slider.bar.w - slider.handle.w;

		// Update the slider value
		slider.value = (slider.handle.x - slider.bar.x) / static_cast<float>(slider.bar.w - slider.handle.w);

		if (!scenarioStarted) {
			return;
		}

		if (slider.label == "Gravity") {
			// Update gravity
			notify(GUI, UPDATE_GRAVITY, getActualSliderValue(slider));
			//m_system->updateGravity(getActualSliderValue(slider));
		}
		else if (slider.label == "Pressure") {
			// Update pressure
			notify(GUI, UPDATE_PRESSURE, getActualSliderValue(slider));
			//m_system->updatePressure(getActualSliderValue(slider));
		}
		else if (slider.label == "Target Density") {
			// Update target density
			notify(GUI, UPDATE_TARGET_DENSITY, getActualSliderValue(slider));
		}
		else if (slider.label == "Viscosity") {
			// Update viscosity
			notify(GUI, UPDATE_VISCOSITY, getActualSliderValue(slider));
			//m_system->updateViscosity(getActualSliderValue(slider));
		}
	}
}

void GUIApplication::handleEvents() {
	SDL_Event e;
	while (SDL_PollEvent(&e) != 0) {
		if (e.type == SDL_QUIT) {
			quit = true;
		}
		else if (e.type == SDL_MOUSEBUTTONDOWN) {
			int x, y;
			SDL_GetMouseState(&x, &y);

			if (!scenarioStarted && isInside(x, y, startButton.x, startButton.y, startButton.w, startButton.h)) {
				std::cout << "Start button clicked" << std::endl;
				this->start();
			}

			if (!scenarioStarted && isInside(x, y, editButton.x, editButton.y, editButton.w, editButton.h)) {
				std::cout << "Edit Configuration button clicked" << std::endl;
				dropdownVisible = !dropdownVisible;
			}

			if (dropdownVisible) {
				int index = handleDropdownClick(x, y, editButton.x, editButton.y + editButton.h, editButton.w, 30, scenarioOptions);
				if (index != -1) {
					selectedScenario = index;
					dropdownVisible = false;
				}
			}

			handleSlider(gravitySlider, x, y);
			handleSlider(pressureSlider, x, y);
			handleSlider(targetDensitySlider, x, y);
			handleSlider(viscositySlider, x, y);

		}
	}
}

void GUIApplication::render() {
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);

	if (!scenarioStarted) {

		renderButton(startButton.x, startButton.y, startButton.w, startButton.h, "Start");

		std::string selectedConfigText = "Select scenario";

		if (selectedScenario != -1) {
			selectedConfigText = "Scenario: " + scenarioOptions[selectedScenario];
		}

		renderButton(editButton.x, editButton.y, editButton.w, editButton.h, selectedConfigText.c_str());

		if (dropdownVisible) {
			renderDropdown(editButton.x, editButton.y + editButton.h, editButton.w, 30, scenarioOptions, selectedScenario);
		}
	}
	else {

		renderSlider(gravitySlider);
		renderSlider(pressureSlider);
		renderSlider(targetDensitySlider);
		renderSlider(viscositySlider);

		// Render FPS and Pressure Square
		renderFPS();
		renderPressureSquare();

	}

	SDL_RenderPresent(renderer);
}

void startLoop(GUIApplication* app) {
	app->run();
}

void GUIApplication::start() {
	/*if (!firstTime) {
		std::terminate();
	}

	firstTime = false;*/

	if (selectedScenario != -1) {
		std::cout << "Starting scenario: " << scenarioOptions[selectedScenario] << std::endl;
	}
	scenarioStarted = true;
	/*delete m_App;
	m_App = new Application(selectedScenario);
	thread = std::thread(startLoop, this);

	m_App->loop();*/
}