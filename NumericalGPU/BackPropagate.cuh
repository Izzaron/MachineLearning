#pragma once

#include "NumericalTypes.cuh"

namespace NumericalGPU {

    int backPropagate(Matrix* dw, Array* db, const Matrix* weights, const Array* biases, const int length, const Array input, const Array output);
}