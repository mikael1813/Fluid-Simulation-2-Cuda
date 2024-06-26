#include <iostream>
#include "Application.hpp"
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

//#include "GUIApplication.hpp"
#include "System.hpp"

using namespace std;

int main(int argc, char* args[]) {

	//Application* app = new Application(0);

	/*GUIApplication gui;
	gui.init();*/

	System system;
	system.init();

	//delete app;

	/*Application app;

	app.loop();
	app.render();

	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
	_CrtDumpMemoryLeaks();*/

	system.thread.join();

	return 0;
}