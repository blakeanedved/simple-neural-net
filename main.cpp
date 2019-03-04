#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include <iomanip>
#include "classes/Input.hpp"
#include "classes/NeuralNet.hpp"

auto Input::Init() -> void {
	this->dataTrainFile.open("datasets/mnist/train/data");
	this->labelTrainFile.open("datasets/mnist/train/labels");
	this->dataTestFile.open("datasets/mnist/test/data");
	this->labelTestFile.open("datasets/mnist/test/labels");

	this->inputShape = 28 * 28;
	this->outputShape = 10;

	this->dataTrainFile.seekg(16, this->dataTrainFile.beg);
	this->labelTrainFile.seekg(8, this->labelTrainFile.beg);
	this->dataTestFile.seekg(16, this->dataTestFile.beg);
	this->labelTestFile.seekg(8, this->labelTestFile.beg);
}

auto Input::NextTrainData() -> std::vector<float> {
	std::vector<float> temp(this->inputShape);
	char tempChar;
	for (int i = 0; i < this->inputShape; i++){
		this->dataTrainFile.read(&tempChar, 1);
		temp[i] = float((unsigned char)tempChar)/255.0;
	}
	return temp;
}

auto Input::NextTrainTarget() -> std::vector<float> {
	char temp;
	this->labelTrainFile.read(&temp, 1);
	std::vector<float> targets(10, 0.0);
	targets[temp] = 1.0;
	return targets;
}

auto Input::NextTestData() -> std::vector<float> {
	std::vector<float> temp(this->inputShape);
	char tempChar;
	for (int i = 0; i < this->inputShape; i++){
		this->dataTestFile.read(&tempChar, 1);
		temp[i] = float((unsigned char)tempChar)/255.0;
	}
	return temp;
}

auto Input::NextTestTarget() -> std::vector<float> {
	char temp;
	this->labelTestFile.read(&temp, 1);
	std::vector<float> targets(10, 0.0);
	targets[temp] = 1.0;
	return targets;
}

auto main() -> int {
	srand(time(0));

	Input input;
	NeuralNet net({ input.inputShape, 784, input.outputShape }, 0.1);

	for (int i = 0; i < 60000; i++){
		net.BackPropagate(input.NextTrainData(), input.NextTrainTarget());
	}

	int correct = 0;
	for (int i = 0; i < 10000; i++){
		net.Fire(input.NextTestData());

		auto outputs = net.GetOutput();
		auto targets = input.NextTestTarget();

		auto outputClass = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
		auto targetClass = std::distance(targets.begin(), std::max_element(targets.begin(), targets.end()));
		if (outputClass == targetClass) correct++;
	}

	std::cout << "Accuracy: " << ((float)correct / 10000.0) * 100.0 << "%" << std::endl;

	return 0;
}
