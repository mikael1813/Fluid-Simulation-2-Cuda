#ifndef OBSERVER_H
#define OBSERVER_H

class Observer {
public:
    virtual ~Observer() {}
    virtual void update(const std::string& app, const std::string& task, const float value) = 0;
};

#endif // OBSERVER_H
