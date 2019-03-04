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

		/** Calculate the derivative of the sigmoid function for a value
		 *
		 * \param val the value to calculate
		 * \ret the calculated value
		 *
		  Simple fast sigmoid derivative function for use of backpropagation
		 */
		inline auto SigmoidDerivative(float val) -> float;

		/** Run the back propagation algorithm, given input and output values
		 *
		 * \param input the input data values
		 * \param target corresponding output data for the input data
		 *
		  Apply the full backpropagation algorithm for a given input and output, which will change the values with no return value
		 */
		auto BackPropagate(std::vector<float> input, std::vector<float> target) -> void;

		/** Apply a simple back propagation algorithm to the deltas of each neurons of each layer
		 *
		 * \param target the target output values for the input data
		 *
		  This function takes in the target values for a piece of data, then calculates the deltas for each neuron in each layer
		 */
		auto BackPropagateDeltas(std::vector<float> target) -> void;

		/** Calculate the adjustment each weight should take
		 *
		 * \ret a 3d vector containing the weight deltas
		 *
		  Calculate how much each weight should be adjusted by factoring its delta and the learning rate of the network, has to return the values instead of setting them to prevent bias from getting an improper adjustment
		 */
		auto BackPropagateWeights() -> std::vector<std::vector<std::vector<float>>>;

		/** Get the accuracy for the network for the last data
		 *
		 * \param target the data with which to determine accuracy with
		 *
		  Calculate the accuracy of the network for given data, using a squared error function
		 */
		auto GetAccuracy(std::vector<float> target) -> float;
};
