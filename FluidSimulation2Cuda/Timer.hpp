#pragma once

#include <chrono>

class Timer {
public:

	static Timer* getInstance() {
		return s_Instance = (s_Instance != nullptr) ? s_Instance : new Timer();
	}

	float getTime();
private:
	static Timer* s_Instance;

	std::chrono::steady_clock::time_point m_lastTime = std::chrono::steady_clock::now();

	Timer() {}
};
