#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <vector>
#include <string>
#include "Application.hpp"

class GUIApplication {
public:
    GUIApplication();
    ~GUIApplication();
    void init();
    void run();

    std::thread thread;

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
    Application* m_App;
    bool firstTime = true;

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

    bool scenarioStarted = false;

    void start();
    void renderButton(int x, int y, int w, int h, const char* text);
    bool isInside(int x, int y, int rectX, int rectY, int rectW, int rectH);
    void renderDropdown(int x, int y, int w, int h, const std::vector<std::string>& options, int selectedIndex);
    int handleDropdownClick(int mouseX, int mouseY, int x, int y, int w, int h, const std::vector<std::string>& options);
    void renderSlider(Slider& slider);
    void handleSlider(Slider& slider, int mouseX, int mouseY);
    void handleEvents();
    void render();
    float GUIApplication::getActualSliderValue(Slider& slider);
};