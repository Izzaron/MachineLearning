#include "ForwardPropagate.cuh"

#include <stdio.h>
#include <math.h>
#include <string>

#define BLOCK_SIZE 16

namespace NumericalGPU {

    __device__ float sigmoidKernel(const float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    __global__ void forwardPropagateKernel(Matrix* weights, Array* biases, Array* z, Array* a, size_t i)
    {
        int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (elementIndex > z[i].length)
            return;

        float zTemp = 0.f;
        for (int row = 0; row < weights[i].width; row++)
        {
            zTemp += weights[i].elements[elementIndex * weights[i].width + row] * a[i].elements[row];
        }
        z[i].elements[elementIndex] = zTemp + biases[i].elements[elementIndex];

        a[i + 1].elements[elementIndex] = sigmoidKernel(z[i].elements[elementIndex]);
    }

    int forwardPropagate(const Matrix* weights, const int weightsLength, const Array* biases, const int biasesLength, const Array* input, Array* output)
    {
        // d_weights
        Matrix* h_weights, * d_weights;
        h_weights = (Matrix*)malloc(weightsLength * sizeof(Matrix));
        for (int i = 0; i < weightsLength; i++)
        {
            h_weights[i].width = weights[i].width;
            h_weights[i].height = weights[i].height;
            cudaMalloc(&h_weights[i].elements, h_weights[i].width * h_weights[i].height * sizeof(float));
            cudaMemcpy(h_weights[i].elements, weights[i].elements, h_weights[i].width * h_weights[i].height * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc(&d_weights, weightsLength * sizeof(Matrix));
        cudaMemcpy(d_weights, h_weights, weightsLength * sizeof(Matrix), cudaMemcpyHostToDevice);

        // d_biases
        Array* h_biases, * d_biases;
        h_biases = (Array*)malloc(biasesLength * sizeof(Array));
        for (int i = 0; i < biasesLength; i++)
        {
            h_biases[i].length = biases[i].length;
            cudaMalloc(&h_biases[i].elements, h_biases[i].length * sizeof(float));
            cudaMemcpy(h_biases[i].elements, biases[i].elements, h_biases[i].length * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc(&d_biases, biasesLength * sizeof(Array));
        cudaMemcpy(d_biases, h_biases, biasesLength * sizeof(Array), cudaMemcpyHostToDevice);


        // d_z
        Array* h_z, * d_z;
        int zLength = biasesLength;
        h_z = (Array*)malloc(zLength * sizeof(Array));
        for (int i = 0; i < zLength; i++)
        {
            h_z[i].length = biases[i].length;
            cudaMalloc(&h_z[i].elements, h_z[i].length * sizeof(float));
        }
        cudaMalloc(&d_z, zLength * sizeof(Array));
        cudaMemcpy(d_z, h_z, zLength * sizeof(Array), cudaMemcpyHostToDevice);
        

        // d_a
        Array* h_a, * d_a;
        int aLength = biasesLength + 1;
        h_a = (Array*)malloc(aLength * sizeof(Array));
        cudaMalloc(&h_a[0].elements, input->length * sizeof(float));
        cudaMemcpy(h_a[0].elements, input->elements, input->length * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 1; i < aLength; i++)
        {
            h_a[i].length = biases[i - 1].length;
            cudaMalloc(&h_a[i].elements, h_a[i].length * sizeof(float));
        }
        cudaMalloc(&d_a, aLength * sizeof(Array));
        cudaMemcpy(d_a, h_a, aLength * sizeof(Array), cudaMemcpyHostToDevice);
        

        for (size_t i = 0; i < biasesLength; i++) {
            //dim3 dimBlock(biases[i].length);
            //dim3 dimGrid(1);
            //forwardPropagateKernel <<< dimGrid, dimBlock >>> (d_weights, d_biases, d_z, d_a, i);
            int GRID_SIZE = (int)ceil(biases[i].length / BLOCK_SIZE);
            forwardPropagateKernel <<< GRID_SIZE, BLOCK_SIZE >>> (d_weights, d_biases, d_z, d_a, i);
            cudaThreadSynchronize();
        }

        // Copy last array of d_a to output
        cudaMemcpy(output->elements, h_a[aLength - 1].elements, output->length * sizeof(float), cudaMemcpyDeviceToHost);

        // Free cuda memory on h_weights elements when kernel is done.
        for (size_t i = 0; i < weightsLength; i++)
        {
            cudaFree(h_weights[i].elements);
        }
        // Free cuda memory d_weights
        cudaFree(d_weights);
        // Free h_weights when kernel is done
        free(h_weights);

        // Free cuda memory on h_biases elements when kernel is done
        for (size_t i = 0; i < biasesLength; i++)
        {
            cudaFree(h_biases[i].elements);
        }
        // Free duca memory d_biases
        cudaFree(d_biases);
        // Free h_biases when kernel is done
        free(h_biases);

        // Free cuda memory on h_z elements when kernel is done
        for (size_t i = 0; i < zLength; i++)
        {
            cudaFree(h_z[i].elements);
        }
        // Free cuda memory d_z
        cudaFree(d_z);
        // Free h_z when kernel is done
        free(h_z);

        // Free cuda memory on h_a elements when kernel is done
        for (size_t i = 0; i < aLength; i++)
        {
            cudaFree(h_a[i].elements);
        }
        // Free cuda memory d_a
        cudaFree(d_a);
        // Free h_a when kernel is done
        free(h_a);

        return 0;
    }
}