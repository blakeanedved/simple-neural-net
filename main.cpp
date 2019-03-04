#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include "classes/Input.hpp"
#include "classes/NeuralNet.hpp"

auto Input::Init() -> void {
	this->inFile.open("datasets/iris/data");
	this->maxData = 150;
	this->inputShape = 4;
	this->outputShape = 3;
	// This is where the first few lines of input should be read for parameters
}

auto Input::NextData() -> std::vector<float> {
	if (this->currentData < this->maxData){
		std::vector<float> temp(4);
		for (int i = 0; i < 4; i++){
			this->inFile >> temp[i];
		}
		return temp;
	} else {
		throw "Out of Data Error";
	}
}

auto Input::NextTarget() -> std::vector<float> {
	if (this->currentData < this->maxData){
		int temp;
		inFile >> temp;
		if (temp == 0){
			return {1.0, 0.0, 0.0};
		} else if (temp == 1){
			return {0.0, 1.0, 0.0};
		} else {
			return {0.0, 0.0, 1.0};
		}
	} else {
		throw "Out of Data Error";
	}
}

auto main() -> int {
	srand(time(0));

	Input input;
	NeuralNet net({ input.inputShape, 100, input.outputShape }, 0.1);

	for (int i = 0; i < 149; i++){
		net.BackPropagate(input.NextData(), input.NextTarget());
	}

	net.Fire(input.NextData());

	auto outputs = net.GetOutput();
	auto targets = input.NextTarget();

	std::cout << "Outputs = { ";
	for (auto output : outputs){
		std::cout << output << " ";
	}
	std::cout << "}" << std::endl << "Targets = { ";
	for (auto target : targets){
		std::cout << target << " ";
	}
	std::cout << "}" << std::endl;

	return 0;
}
