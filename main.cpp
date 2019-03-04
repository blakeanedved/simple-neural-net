#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <random>
#include "classes/NeuralNet.hpp"

std::ifstream inFile;

auto getNextData() -> std::vector<float> {
	std::vector<float> temp(4);
	for (int i = 0; i < 4; i++){
		inFile >> temp[i];
	}
	return temp;
}

auto getNextTarget() -> std::vector<float> {
	int temp;
	inFile >> temp;
	if (temp == 0){
		return {1.0, 0.0, 0.0};
	} else if (temp == 1){
		return {0.0, 1.0, 0.0};
	} else {
		return {0.0, 0.0, 1.0};
	}
}

auto main() -> int {
	// Seed the random number generator for when the layers create random neurons
	srand(time(0));

	inFile.open("datasets/iris/data");

	// Instantiate a NeuralNet with 3 layers, with 5 neurons in each, and a learning rate of 0.1
	NeuralNet net({ 4, 3 }, 0.1);

	for (int i = 0; i < 149; i++){
		net.BackPropagate(getNextData(), getNextTarget());
	}

	net.Fire(getNextData());

	auto outputs = net.GetOutput();
	auto targets = getNextTarget();

	std::cout << "Outputs = { ";
	for (auto output : outputs){
		std::cout << output << " ";
	}
	std::cout << "}" << std::endl << "Targets = { ";
	for (auto target : targets){
		std::cout << target << " ";
	}
	std::cout << "}" << std::endl;

	inFile.close();

	return 0;
}
