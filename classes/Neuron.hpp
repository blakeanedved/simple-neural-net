#ifndef _SNN_CLASS_NEURON_HPP_
#define _SNN_CLASS_NEURON_HPP_
#include <vector>

/** A functional neuron with the ability to calculate its own value
 *
  This class supplies the functions necessary for basic neuron activity, such as firing and resetting, and comes packaged with a normalization function
 */
class Neuron {
	public:
		std::vector<float> weights; /**< Vector to hold the weights for each incoming value */

		float value; /**< This neurons ouput after being fired */
		float bias; /**< The extra value added during the firing of a neuron */

		/** Constructor with only the number of input neurons
		 *
		 * \param numInputNeurons the number of input neurons this neuron should generate weights for
		 *
		  A constructor that builds the weights for this neuron randomly, and sets the bias randomly to have a basic starting point to build off
		 */
		Neuron(int numInputNeurons);

		/** Constructor with the number of input neurons, a weight initializer, and a bias value
		 *
		 * \param numInputNeurons the number of neurons to use the weight initializer to generate
		 * \param initializedWeights a vector containing the inital weights for this neuron
		 * \param initializedBias the value to set this neurons bias to
		 * 
		  A constructor that builds the neuron using supplied values so that nothing is random
		 */
		Neuron(int numInputNeurons, std::vector<float> initializedWeights, float initializedBias);
		
		/** A method to get the ouput value of the neuron
		 *
		 * \param &inputValues a reference to the output from the last layerin the form of a float vector
		 *
		  This method takes a reference to the outputs of the previous layer and multiplies them by its own weight to sum and normalize into a value stored at `this->value`
		 */
		auto Fire(std::vector<float> &inputValues) -> void;

		/** Reset the value of this neuron
		 *
		  This method resets the value of this neuron and lays a method to reset more parameters in the future
		 */
		auto Reset() -> void;

		/** Normalize this neurons value
		 *
		 * \param val the value to normalize
		 * \return The normalized value
		 *
		  This method uses the sigmoid function to normalize the output value of this neuron
		 */
		auto Sigmoid(float) -> float;
};

#endif
