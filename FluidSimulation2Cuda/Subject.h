#ifndef SUBJECT_H
#define SUBJECT_H

#include "Observer.h"
#include <vector>
#include <algorithm>

class Subject {
public:
	virtual ~Subject() {}
	void attach(Observer* observer) {
		observers.push_back(observer);
	}
	void detach(Observer* observer) {
		observers.erase(std::remove(observers.begin(), observers.end(), observer), observers.end());
	}
	void notify(const std::string& app, const std::string& task, const float value) {
		for (Observer* observer : observers) {
			observer->update(app, task, value);
		}
	}

private:
	std::vector<Observer*> observers;
};

#endif // SUBJECT_H
