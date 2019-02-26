#include <vector>
#include "Layer.hpp"

/** A class to perform the main functions for a neural network
 *
  This class holds a vector of neurons and the methods to act upon them, including a Fire method to get the output of the entire network
 */
class NeuralNet {
	private:
		std::vector<Layer> layers; /**< Vector of layer that this network acts upon */

		float learningRate; /**< A value to describe how fast the weights change for this network */

	public:

		/** Make the network, with a vector containing the amount of neurons in each layer, and the learning rate
		 *
		 * \param numNeuronsPerLayer A vector containing the amount of neurons to put in each layer, in order
		 * \param learningRate a value to describe how fast the weights for this network change
		 *
		  This constructor builds a neural net, fully functional, with random values for each neuron as a starting point.
		 */
		NeuralNet(std::vector<int> numNeuronsPerLayer, float learningRate);

		/** Reset the values in this network
		 *
		  This function iteratively resets the values in each of the layers this net contains
		 */
		auto Reset() -> void;

		/** Activate the neural net on data
		 *
		 * \param input the vector of input values to give to the first layer of this network
		 *
		  This function will loop through the entire network, feeding each layer's output to the next layer, until it gets to the end with a predictions on the data
		 */
		auto Fire(std::vector<float> input) -> void;

		/** Return the output of the network
		 *
		 * \ret a vector containing the output values of the last layer in this network
		 *
		  This function just grabs the values from the last layer of the network
		 */
		auto GetOutput() -> std::vector<float>;
};
