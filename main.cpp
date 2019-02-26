#include <iostream>
#include <vector>
#include <time.h>
#include "classes/NeuralNet.hpp"

auto main() -> int {
	// Seed the random number generator for when the layers create random neurons
	srand(time(0));

	// Instantiate a NeuralNet with 3 layers, with 5 neurons in each, and a learning rate of 0.1
	NeuralNet net({ 5, 5, 5 }, 0.1);

	// Activate the net with the input values 1.0, 2.0, 3.0, 4.0, 5.0
	net.Fire({1.0, 2.0, 3.0, 4.0, 5.0});

	// Get the outputs of the NeuralNet
	auto outputs = net.GetOutput();

	// Loop through the ouputs of the NeuralNet and print them to stdout on newlines
	for (auto output : outputs){
		std::cout << output << std::endl;
	}

	return 0;
}
