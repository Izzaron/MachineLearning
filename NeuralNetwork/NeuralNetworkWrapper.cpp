#include "neuralNetwork.hpp"
#include "exampleDatabase.hpp"
#include <iostream>
#include <chrono>
#include <thread>

extern "C" __declspec(dllexport) NeuralNetwork* CreateNeuralNetwork(const int* layerLengths, const int nrOfLayers) {
	return new NeuralNetwork(std::vector<int>(layerLengths,layerLengths+nrOfLayers));
}

extern "C" __declspec(dllexport) void DeleteNeuralNetwork(NeuralNetwork* neuralNetwork) {
	delete neuralNetwork;
}

extern "C" __declspec(dllexport) int LoadNeuralNetworkWeights(NeuralNetwork* neuralNetwork) {
	return (int)neuralNetwork->loadWeights();
}

extern "C" __declspec(dllexport) int MakeNeuralNetworkPrediction(NeuralNetwork* neuralNetwork, float* input, const int inputLength, float* output, const int outputLength) {
	
	// Verify input lengths
	if (inputLength != neuralNetwork->getLayerLengths().front()) {
		std::cout << "input lengths dont agree" << std::endl;
		return false;
	}
	if (outputLength != neuralNetwork->getLayerLengths().back()) {
		std::cout << "output lengths dont agree" << std::endl;
		return false;
	}

	// Convert to Eigen Vector
	Eigen::VectorXf input_vector = Eigen::VectorXf::Map(input, inputLength);

	// Run guess
	Eigen::VectorXf output_vector = neuralNetwork->forwardPropagate(input_vector);

	// Copy results to output array
	memcpy(output, output_vector.data(), outputLength * sizeof(*output));

	return true;
}

extern "C" __declspec(dllexport) int TrainOnMnist(NeuralNetwork* neuralNetwork, char* images_filepath, char* labels_filepath, int nrOfEpochs) {
	
	if (neuralNetwork->getLayerLengths().front() != 28*28) {
		std::cout << "This Network cannot train on MNIST" << std::endl;
		std::cout << "input length not 28*28" << std::endl;
		return false;
	}
	if (neuralNetwork->getLayerLengths().back() != 10) {
		std::cout << "This Network cannot train on MNIST" << std::endl;
		std::cout << "output length not 10" << std::endl;
		return false;
	}
	
	ExampleDatabase ed;
	ed.readMnistImages(images_filepath, 60000);
	ed.readMnistLabels(labels_filepath);

	std::vector<const example*> allExamples = ed.getAllExamples();

	const auto processor_count = std::thread::hardware_concurrency();

	if (processor_count == 0) {
		neuralNetwork->train(allExamples, 0.25f, 8, nrOfEpochs);
	}
	else {
		neuralNetwork->train(allExamples, 0.25f, processor_count - 1, nrOfEpochs);
	}

	return true;
}

//This is to be removed. Its merely kept for testing
extern "C" __declspec(dllexport) void AddWithCuda(NeuralNetwork * neuralNetwork) {

	// Test backPropagate
	std::cout << "testing backPropagate" << std::endl;

	Eigen::VectorXf input(28*28);
	input.setOnes();
	Eigen::VectorXf output(10);
	output.setOnes();

	auto startBP = std::chrono::high_resolution_clock::now();
	auto rtn1 = neuralNetwork->backPropagate(input,output);
	auto middleBP = std::chrono::high_resolution_clock::now();
	auto rtn2 = neuralNetwork->backPropagateGPU(input, output);
	auto endBP = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> durationCPUBP = middleBP - startBP;
	std::chrono::duration<double> durationGPUBP = endBP - middleBP;

	std::cout << "CPU(" << durationCPUBP.count() << ") " << std::endl;
	std::cout << "GPU(" << durationGPUBP.count() << ") " << std::endl;
	for (size_t i = 0; i < rtn1.biases.size(); i++)
	{
		std::cout <<  rtn1.biases[i].transpose() << std::endl;
		std::cout <<  rtn2.biases[i].transpose() << std::endl;
	}

	// Test forwardPropagate
	std::cout << "testing forwardPropagate" << std::endl;

	auto startFP = std::chrono::high_resolution_clock::now();
	auto rtn3 = neuralNetwork->forwardPropagate(input);
	auto middleFP = std::chrono::high_resolution_clock::now();
	auto rtn4 = neuralNetwork->forwardPropagateGPU(input);
	auto endFP = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> durationCPUFP = middleFP - startFP;
	std::chrono::duration<double> durationGPUFP = endFP - middleFP;

	std::cout << "CPU(" << durationCPUFP.count() << "): " << rtn3.transpose() << std::endl;
	std::cout << "GPU(" << durationGPUFP.count() << "): " << rtn4.transpose() << std::endl;

}

// To be added (a more general training interface):
//extern "C" __declspec(dllexport) void Train(NeuralNetwork * neuralNetwork, std::vector<float[]> inputs, std::vector<float[]> answers, int nrOfEpochs);