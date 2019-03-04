main:main.cpp
	g++ main.cpp classes/NeuralNet.cpp classes/Layer.cpp classes/Neuron.cpp -std=c++17 -O2 -lpthread

debug:main.cpp
	g++ -g main.cpp classes/NeuralNet.cpp classes/Layer.cpp classes/Neuron.cpp -std=c++17 -O2 -lpthread

docs:
	doxygen Doxygen/Doxyfile

all:main
