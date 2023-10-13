#pragma once

#include "NumericalTypes.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace NumericalGPU {

    __global__ void forwardPropagateKernel(Matrix* weights, Array* biases, Array* z, Array* a, size_t i);
    __device__ float sigmoidKernel(const float x);
    int forwardPropagate(const Matrix* weights, const int weightsLength, const Array* biases, const int biasesLength, const Array* input, Array* output);
}