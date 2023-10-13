#pragma once

#include <Eigen/Dense>
//#include "example.hpp"
#include "asyncReturnData.hpp"

typedef std::pair<Eigen::VectorXf, Eigen::VectorXf> example;
class NeuralNetwork
{
private:

    // Neural layer lengths
    std::vector<int>                layerLengths;

    // Weights
    std::vector<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >    weights;

    // Biases
    std::vector<Eigen::VectorXf>    biases;
public:
    //Private functions
    AsyncReturnData                 backPropagate               (const Eigen::VectorXf& input, const Eigen::VectorXf& output);
    AsyncReturnData                 backPropagateGPU            (const Eigen::VectorXf& input, const Eigen::VectorXf& output);
    void                            backPropagateBatch          (std::vector< const example* > batch, float eta);

//Public: is supposed to be here

    //Constructors
    NeuralNetwork(std::vector<int> layerLengths);
    ~NeuralNetwork();

    const Eigen::VectorXf           forwardPropagate            (const Eigen::VectorXf & input);
    const Eigen::VectorXf           forwardPropagateGPU         (const Eigen::VectorXf& input);
    void                            train                       (std::vector< const example* > &examples, float eta, int batchSize, int nrOfEpochs);
    bool                            loadWeights                 ();
    void                            writeWeights                ();
    const std::vector<int>          getLayerLengths             ();
};