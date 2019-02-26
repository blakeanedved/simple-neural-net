#ifndef _SNN_CLASSES_LAYER_HPP_
#define _SNN_CLASSES_LAYER_HPP_
#include <vector>
#include "Neuron.hpp"

/** A container to hold neurons, and provide functions for them to process
 *
  This class supplies the methods to provide Neurons the ability to think collectively off of a reference to a previous layer's values
 */
class Layer {
	public:
		std::vector<Neuron> neurons; /**< Vector to hold the neurons in this layer */

		/** Constructor with the number of neurons and the number of input neurons
		 *
		 * \param numOfNeurons the number of neurons that will be in this layer
		 * \param numInputNeurons the number of neurons in the previus layer
		 *
		  This constructor builds this layer using the previous layer's neurons to make each neuron, along with the amount of neurons in this layer
		 */
		Layer(int numOfNeurons, int numInputNeurons);

		/** Constructor with the number of neurons
		 *
		 * \param numOfNeurons the number of neurons that will be in this layer
		 *
		  This constructor builds this layer using only the number of neurons, used mostly as a constructor for the first layer
		 */
		Layer(int numOfNeurons);

		/** Compute this layers values with reference to the previous layers values
		 *
		 * \param inputValues a reference to the previous layers values
		 *
		  This method will iteratively call each neurons .Fire() method to compute the results for this layer
		 */
		auto Fire(std::vector<float> &inputValues) -> void;

		/** Reset the value of each neuron
		 *
		  This method iteratively sets the value to 0.0f for each neuron, used as a setup to another test of the network
		 */
		auto Reset() -> void;

		/** Get this layers values
		 *
		 * \ret a vector containing this layers values
		  This method gets the value from each neuron, puts them into a vector and returns it
		 */
		auto GetValues() -> std::vector<float>;
};

#endif
