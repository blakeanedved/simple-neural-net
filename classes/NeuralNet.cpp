#include <iostream>
#include <vector>
#include "NeuralNet.hpp"
//#include "Layer.cpp"

NeuralNet::NeuralNet(std::vector<int> numNeuronsPerLayer, float learningRate){
	
	for (int i = 0; i < numNeuronsPerLayer.size(); i++){
		if (i == 0) this->layers.push_back({numNeuronsPerLayer[i]});
		else this->layers.push_back({numNeuronsPerLayer[i], numNeuronsPerLayer[i - 1]});
	}

	this->learningRate = learningRate;
}

auto NeuralNet::Reset() -> void {
	for (auto &layer : this->layers){
		layer.Reset();
	}
}

auto NeuralNet::Fire(std::vector<float> input) -> void {
	this->Reset();

	for (int i = 0; i < this->layers[0].neurons.size(); i++){
		this->layers[0].neurons[i].value = input[i];
	}
	for (int i = 1; i < this->layers.size(); i++){
		std::vector<float> lastValues = this->layers[i - 1].GetValues();
		this->layers[i].Fire(lastValues);
	}
}

auto NeuralNet::GetOutput() -> std::vector<float> {
	return this->layers[this->layers.size() - 1].GetValues();
}
