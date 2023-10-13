#include <iostream>
#include <random>
#include <future>
#include <filesystem>
#include "neuralNetwork.hpp"
#include "eigenIO.hpp"
#include "NumericalGPU.hpp"

NeuralNetwork::NeuralNetwork(std::vector<int> layerLengths): layerLengths(layerLengths)
{
    // Initialize weights and biases
    for (int i = 1; i < layerLengths.size(); i++)
    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& w = weights.emplace_back(layerLengths[i],layerLengths[i-1]);
        w.setRandom();
        
        Eigen::VectorXf& b = biases.emplace_back(layerLengths[i]);
        b.setRandom();
    }
}

NeuralNetwork::~NeuralNetwork()
{
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoidPrime(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
}

const Eigen::VectorXf NeuralNetwork::forwardPropagate(const Eigen::VectorXf &input)
{
    if (input.size() != layerLengths.front())
        throw std::invalid_argument("input needs to be same length as first layer");

    std::vector<Eigen::VectorXf> z(layerLengths.size()-1);
    std::vector<Eigen::VectorXf> a(layerLengths.size());

    a[0] = input;
    for (size_t i = 0; i < layerLengths.size()-1; i++){
        z[i] = weights[i]*a[i] + biases[i];
        a[i+1] = z[i].unaryExpr(std::ref(sigmoid));
    }    

    return a.back();
}

const Eigen::VectorXf NeuralNetwork::forwardPropagateGPU(const Eigen::VectorXf& input)
{
    if (input.size() != layerLengths.front())
        throw std::invalid_argument("input needs to be same length as first layer");

    NumericalGPU::Matrix *h_weights = (NumericalGPU::Matrix*)malloc(weights.size() * sizeof(NumericalGPU::Matrix));
    for (size_t i = 0; i < weights.size(); i++)
    {
        h_weights[i].width = weights[i].cols();
        h_weights[i].height = weights[i].rows();
        h_weights[i].elements = (float*)malloc(h_weights[i].width * h_weights[i].height * sizeof(float));
        memcpy(h_weights[i].elements, weights[i].data(), h_weights[i].width * h_weights[i].height * sizeof(float));
    }

    NumericalGPU::Array *h_biases = (NumericalGPU::Array*)malloc(biases.size() * sizeof(NumericalGPU::Array));
    for (size_t i = 0; i < biases.size(); i++)
    {
        h_biases[i].length = biases[i].size();
        h_biases[i].elements = (float*)malloc(h_biases[i].length * sizeof(float));
        memcpy(h_biases[i].elements, biases[i].data(), h_biases[i].length * sizeof(float));
    }

    NumericalGPU::Array h_input;
    h_input.length = input.size();
    h_input.elements = (float*)malloc(h_input.length * sizeof(float));
    memcpy(h_input.elements, input.data(), h_input.length * sizeof(float));

    NumericalGPU::Array h_output;
    h_output.length = layerLengths.back();
    h_output.elements = (float*)malloc(h_output.length * sizeof(float));

    NumericalGPU::forwardPropagate(h_weights, weights.size(), h_biases, biases.size(), &h_input, &h_output);

    Eigen::VectorXf ret = Eigen::VectorXf::Map(h_output.elements, h_output.length);
    
    // Free memory
    for (size_t i = 0; i < weights.size(); i++) {
        free(h_weights[i].elements);
    }
    free(h_weights);
    for (size_t i = 0; i < weights.size(); i++) {
        free(h_biases[i].elements);
    }
    free(h_biases);
    free(h_input.elements);
    free(h_output.elements);

    return ret;
}

AsyncReturnData NeuralNetwork::backPropagate(const Eigen::VectorXf& input,const Eigen::VectorXf& output)
{
    if (input.size() != layerLengths.front())
        throw std::invalid_argument("The input vector needs to be same length as first layer");

    if (output.size() != layerLengths.back())
        throw std::invalid_argument("The output vector needs to be same length as last layer");

    // Forward propagation
    std::vector<Eigen::VectorXf> z(layerLengths.size()-1);
    std::vector<Eigen::VectorXf> a(layerLengths.size());

    a[0] = input;
    for (size_t i = 0; i < layerLengths.size()-1; i++){
        z[i] = weights[i]*a[i] + biases[i];
        a[i+1] = z[i].unaryExpr(std::ref(sigmoid));
    }

    // Back propagation
    std::vector<Eigen::VectorXf> d(layerLengths.size()-1);
    
    // Last layer
    d.back() = (a.back() - output).array() * z.back().unaryExpr(std::ref(sigmoidPrime)).array();

    // Remaining layers
    for (size_t i = layerLengths.size()-2; i > 0; i--)
    {
        d[i-1] = (weights[i].transpose() * d[i]).array() * z[i-1].unaryExpr(std::ref(sigmoidPrime)).array();
    }

    // Return results
    AsyncReturnData rtn = AsyncReturnData((int)layerLengths.size()-1);

    for (size_t i = 0; i < rtn.layerCount; i++)
    {
        rtn.weights[i] =    d[i] * a[i].transpose();
        rtn.biases [i] =    d[i]                   ;
    }

    return rtn;
}

AsyncReturnData NeuralNetwork::backPropagateGPU(const Eigen::VectorXf& input, const Eigen::VectorXf& output)
{
    if (input.size() != layerLengths.front())
        throw std::invalid_argument("The input vector needs to be same length as first layer");

    if (output.size() != layerLengths.back())
        throw std::invalid_argument("The output vector needs to be same length as last layer");

    NumericalGPU::Matrix* h_weights = (NumericalGPU::Matrix*)malloc(weights.size() * sizeof(NumericalGPU::Matrix));
    for (size_t i = 0; i < weights.size(); i++)
    {
        h_weights[i].width = weights[i].cols();
        h_weights[i].height = weights[i].rows();
        h_weights[i].elements = (float*)malloc(h_weights[i].width * h_weights[i].height * sizeof(float));
        memcpy(h_weights[i].elements, weights[i].data(), h_weights[i].width * h_weights[i].height * sizeof(float));
    }

    NumericalGPU::Array* h_biases = (NumericalGPU::Array*)malloc(biases.size() * sizeof(NumericalGPU::Array));
    for (size_t i = 0; i < biases.size(); i++)
    {
        h_biases[i].length = biases[i].size();
        h_biases[i].elements = (float*)malloc(h_biases[i].length * sizeof(float));
        memcpy(h_biases[i].elements, biases[i].data(), h_biases[i].length * sizeof(float));
    }

    NumericalGPU::Array h_input;
    h_input.length = input.size();
    h_input.elements = (float*)malloc(h_input.length * sizeof(float));
    memcpy(h_input.elements, input.data(), h_input.length * sizeof(float));

    NumericalGPU::Array h_output;
    h_output.length = layerLengths.back();
    h_output.elements = (float*)malloc(h_output.length * sizeof(float));
    memcpy(h_output.elements, output.data(), h_output.length * sizeof(float));

    AsyncReturnData rtn = AsyncReturnData((int)layerLengths.size() - 1);
    NumericalGPU::Matrix* dw = (NumericalGPU::Matrix*)malloc(weights.size() * sizeof(NumericalGPU::Matrix));
    NumericalGPU::Array* db = (NumericalGPU::Array*)malloc(biases.size() * sizeof(NumericalGPU::Array));
    for (size_t i = 0; i < rtn.layerCount; i++)
    {
        dw[i].width = weights[i].cols();
        dw[i].height = weights[i].rows();
        rtn.weights[i] = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
        rtn.weights[i].resize(weights[i].rows(), weights[i].cols());
        dw[i].elements = rtn.weights[i].data();

        db[i].length = biases[i].size();
        rtn.biases[i].resize(biases[i].size());
        db[i].elements = rtn.biases[i].data();
    }
    
    NumericalGPU::backPropagate(dw,db,h_weights, h_biases, layerLengths.size()-1, h_input, h_output);

    // Free memory
    for (size_t i = 0; i < weights.size(); i++) {
        free(h_weights[i].elements);
    }
    free(h_weights);
    for (size_t i = 0; i < weights.size(); i++) {
        free(h_biases[i].elements);
    }
    free(h_biases);
    free(h_input.elements);
    free(h_output.elements);
    free(dw);
    free(db);

    return rtn;
}

void NeuralNetwork::backPropagateBatch(std::vector< const example* > batch, float eta)
{
    // Delta weights
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >        dW = weights;
    for (auto& w : dW)
        w.setZero();

    // Delta biases
    std::vector<Eigen::VectorXf>        db = biases;
    for (auto& b : db)
        b.setZero();

    // Start the whole batch asynchronically
    std::vector<std::future<AsyncReturnData> > futures;

    for (auto example : batch)
    {
        std::future<AsyncReturnData> ftr = std::async(
            std::launch::async,
            &NeuralNetwork::backPropagate,
            this,
            example->first,
            example->second
        );

        futures.push_back(std::move(ftr));
    }

    // Collect the threads
    for (auto&& future : futures)
    {
        // Wait for each thread
        auto rtn = future.get();

        // Accumulate the results
        for (size_t i = 0; i < rtn.layerCount; i++)
        {
            dW[i] += rtn.weights[i];
            db[i] += rtn.biases[i];
        }
    }

    // Update all weights
    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] -= dW[i] * eta / batch.size();
        biases[i] -= db[i] * eta / batch.size();
    }
}

void NeuralNetwork::train (std::vector< const example* > &examples, float eta, int batchSize, int nrOfEpochs)
{
    std::chrono::steady_clock::time_point clock;

    for (size_t i = 0; i < nrOfEpochs; i++)
    {
        clock = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch: " << i+1 << "/" << nrOfEpochs << std::endl;

        auto rng = std::default_random_engine {};

        shuffle(begin(examples),end(examples),rng);

        size_t nrOfBatches = examples.size()/batchSize;

        for (size_t i = 0; i < nrOfBatches; i++)
        {
            backPropagateBatch({examples.begin() + i*batchSize, examples.begin() + (i+1)*batchSize}, eta);
        }

        std::cout << "Epoch " << i+1 << " took " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - clock).count() << " seconds." << std::endl;

    }

    this->writeWeights();
    
}

const std::vector<int> NeuralNetwork::getLayerLengths()
{
    return layerLengths;
}

bool NeuralNetwork::loadWeights()
{
    for (size_t i = 0; i < layerLengths.size()-1; i++)
    {
        // Weights
        std::string weightPath = "weights\\W" + std::to_string(i+1) + ".mat";
        if (!std::filesystem::exists(weightPath)) {
            std::cout << "\"" << weightPath << "\" not found!" << std::endl;
            return false;
        }
        EigenIO::read_binary(weightPath, weights[i]);
        
        // Biases
        std::string biasPath = "weights\\b" + std::to_string(i+1) + ".mat";
        if (!std::filesystem::exists(biasPath)) {
            std::cout << "\"" << biasPath << "\" not found!" << std::endl;
            return false;
        }
        EigenIO::read_binary(biasPath, biases[i]);
    }
    return true;
}

void NeuralNetwork::writeWeights()
{
    std::cout << "Writing weights..." << std::endl;

    std::filesystem::create_directory("weights");

    for (size_t i = 0; i < layerLengths.size()-1; i++)
    {
        // Weights
        std::string weightPath = "weights\\W" + std::to_string(i+1) + ".mat";
        EigenIO::write_binary(weightPath, weights[i]);
        
        // Biases
        std::string biasPath = "weights\\b" + std::to_string(i+1) + ".mat";
        EigenIO::write_binary(biasPath, biases[i]);
    }

    std::cout << "Weights written." << std::endl;
}