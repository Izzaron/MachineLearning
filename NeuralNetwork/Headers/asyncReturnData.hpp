#pragma once

#include <vector>
#include <Eigen/Dense>

struct AsyncReturnData
{

public:
    int                                                                             layerCount;
    std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> >       weights;
    std::vector< Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> >       biases;    

    AsyncReturnData() = default; //Added just so it will compile. Probably wrong. Not needed on g++
    AsyncReturnData(int nrOfLayers);

    ~AsyncReturnData();
};