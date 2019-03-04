#include <iostream>
#include <vector>
#include <math.h>
#include "Neuron.hpp"

Neuron::Neuron(int numInputNeurons){
	for (int i = 0; i < numInputNeurons; i++){
		this->weights.push_back(float(rand() % 100) / 100 - 0.5);
	}

	this->bias = float(rand() % 100) / 100;
	this->value = 0;
}

Neuron::Neuron(int numInputNeurons, std::vector<float> initializedWeights, float initializedBias){
	if (initializedWeights.size() != numInputNeurons){
		std::cerr << "Unmatching weight initializer and input neurons" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numInputNeurons; i++){
		this->weights.push_back(initializedWeights[i]);
	}

	this->bias = initializedBias;
	this->value = 0;
}

auto Neuron::Sigmoid(float val) -> float {
	return 1.0 / (1.0 + exp(-val));
}

auto Neuron::Fire(std::vector<float> &inputValues) -> void {
	if (inputValues.size() != this->weights.size()){
		std::cerr << "Unmatching previous layer with weights" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < inputValues.size(); i++){
		this->value += inputValues[i] * this->weights[i];
	}

	// if using bias
	// this->value += this->bias;
	this->valuePreSigmoid = this->value;
	this->value = this->Sigmoid(this->value);
}

auto Neuron::Reset() -> void {
	this->value = 0;
}
