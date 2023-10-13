#pragma once

namespace NumericalGPU {

    typedef struct {
        int width;
        int height;
        float* elements;
    } Matrix;

    typedef struct {
        int length;
        float* elements;
    } Array;
}