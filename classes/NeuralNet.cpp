#include <iostream>
#include <vector>
#include <math.h>
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

inline auto NeuralNet::SigmoidDerivative(float val) -> float {
	return 0.5 / ((1.0f + fabs(val)) * (1.0f + fabs(val)));
}

auto NeuralNet::BackPropagate(std::vector<float> input, std::vector<float> target) -> void {
	this->Fire(input);

	this->BackPropagateDeltas(target);
	auto weightDeltas = this->BackPropagateWeights();

	for (int i = 1; i < this->layers.size(); i++){
		for (int j = 0; j < this->layers[i].neurons.size(); j++){
			for (int k = 0; k < this->layers[i].neurons[j].weights.size(); k++){
				this->layers[i].neurons[j].weights[k] += weightDeltas[i][j][k];
			}
		}
	}
}

auto NeuralNet::BackPropagateDeltas(std::vector<float> target) -> void {
	for (int i = this->layers.size() - 1; i > 0; i--){
		for (int j = 0; j < this->layers[i].neurons.size(); j++){
			if (i == this->layers.size() - 1){
				this->layers[i].neurons[j].delta = 2*(this->layers[i].neurons[j].value - target[j]);
			} else {
				this->layers[i].neurons[j].delta = 0;

				for (int k = 0; k < this->layers[i + 1].neurons.size(); k++){
					this->layers[i].neurons[j].delta += this->layers[i + 1].neurons[k].delta * this->SigmoidDerivative(this->layers[i + 1].neurons[k].valuePreSigmoid) * this->layers[i + 1].neurons[k].weights[j];
				}
			}
		}
	}
}

auto NeuralNet::BackPropagateWeights() -> std::vector<std::vector<std::vector<float>>> {
	std::vector<std::vector<std::vector<float>>> weightDeltas(this->layers.size());

	for (int i = 1; i < this->layers.size(); ++i){
		weightDeltas[i] = std::vector<std::vector<float>>(this->layers[i].neurons.size());
		for (int j = 0; j < this->layers[i].neurons.size(); j++){
			weightDeltas[i][j] = std::vector<float>(this->layers[i-1].neurons.size());
		}
	}

	for (int i = this->layers.size() - 1; i > 0; i--){
		for (int j = 0; j < this->layers[i].neurons.size(); j++){
			for (int k = 0; k < this->layers[i].neurons[j].weights.size(); k++){
				weightDeltas[i][j][k] = -this->learningRate*this->layers[i].neurons[j].delta*this->SigmoidDerivative(this->layers[i].neurons[j].valuePreSigmoid)*this->layers[i - 1].neurons[k].value;
			}
		}
	}

	return weightDeltas;
}

auto NeuralNet::GetAccuracy(std::vector<float> target) -> float {
	float totalError = 0.0f;

	for (int i = 0; i < target.size(); i++){
		totalError += 0.5*(target[i] - this->layers[this->layers.size() - 1].neurons[i].value)*(target[i] - this->layers[this->layers.size() - 1].neurons[i].value);
	}

	return totalError;
}
