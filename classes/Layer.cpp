#include <iostream>
#include <vector>
#include "Layer.hpp"

Layer::Layer(int numOfNeurons, int numInputNeurons){
	for (int i = 0; i < numOfNeurons; i++){
		this->neurons.push_back({numInputNeurons});
	}
}

Layer::Layer(int numOfNeurons){
	std::vector<float> zeroWeights(numOfNeurons, 0.0f);

	for (int i = 0; i < numOfNeurons; i++){
		zeroWeights[i] = 1.0f;
		this->neurons.push_back({numOfNeurons});
		zeroWeights[i] = 0.0f;
	}
}

auto Layer::Fire(std::vector<float> &inputValues) -> void {
	for (auto &neuron : this->neurons){
		neuron.Fire(inputValues);
	}
}

auto Layer::Reset() -> void {
	for (auto &neuron : this->neurons){
		neuron.Reset();
	}
}

auto Layer::GetValues() -> std::vector<float> {
	std::vector<float> values;
	for (auto neuron : this->neurons){
		values.push_back(neuron.value);
	}
	return values;
};
