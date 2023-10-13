#include "BackPropagate.cuh"
#include "ForwardPropagate.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <string>
#include <cassert>

#define BLOCK_SIZE 16

namespace NumericalGPU {

    __device__ float sigmoidPrimeKernel(const float x)
    {
        return sigmoidKernel(x) * (1.0f - sigmoidKernel(x));
    }

    __global__ void lastLayer(Array d, const Array a, const Array output, const Array z)
    {
        int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (elementIndex >= d.length)
            return;

        d.elements[elementIndex] = (a.elements[elementIndex] - output.elements[elementIndex]) * sigmoidPrimeKernel(z.elements[elementIndex]);
    }

    __global__ void remainingLayers(Array* d, const Matrix* weights, const Array* z, size_t i)
    {
        int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (elementIndex >= d[i - 1].length)
            return;

        float dTemp = 0.f;
        for (int row = 0; row < weights[i].width; row++)
        {
            dTemp += weights[i].elements[row * weights[i].width + elementIndex] * d[i].elements[row];
        }

        d[i - 1].elements[elementIndex] = dTemp * sigmoidPrimeKernel(z[i - 1].elements[elementIndex]);
    }

    __global__ void formWeightsUpdate(Matrix dw, const Array d, const Array a)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if ((row >= dw.height) || (col >= dw.width))
            return;

        dw.elements[row * dw.width + col] = d.elements[row] * a.elements[col];
    }

    int backPropagate(Matrix* dw, Array* db, const Matrix* weights, const Array* biases, const int length, const Array input, const Array output)
    {
        // d_weights
        Matrix* h_weights, * d_weights;
        h_weights = (Matrix*)malloc(length * sizeof(Matrix));
        for (int i = 0; i < length; i++)
        {
            h_weights[i].width = weights[i].width;
            h_weights[i].height = weights[i].height;
            cudaMalloc(&h_weights[i].elements, h_weights[i].width * h_weights[i].height * sizeof(float));
            cudaMemcpy(h_weights[i].elements, weights[i].elements, h_weights[i].width * h_weights[i].height * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc(&d_weights, length * sizeof(Matrix));
        cudaMemcpy(d_weights, h_weights, length * sizeof(Matrix), cudaMemcpyHostToDevice);

        // d_biases
        Array* h_biases, * d_biases;
        h_biases = (Array*)malloc(length * sizeof(Array));
        for (int i = 0; i < length; i++)
        {
            h_biases[i].length = biases[i].length;
            cudaMalloc(&h_biases[i].elements, h_biases[i].length * sizeof(float));
            cudaMemcpy(h_biases[i].elements, biases[i].elements, h_biases[i].length * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc(&d_biases, length * sizeof(Array));
        cudaMemcpy(d_biases, h_biases, length * sizeof(Array), cudaMemcpyHostToDevice);

        // d_z
        Array* h_z, * d_z;
        int zLength = length;
        h_z = (Array*)malloc(zLength * sizeof(Array));
        for (int i = 0; i < zLength; i++)
        {
            h_z[i].length = biases[i].length;
            cudaMalloc(&h_z[i].elements, h_z[i].length * sizeof(float));
        }
        cudaMalloc(&d_z, zLength * sizeof(Array));
        cudaMemcpy(d_z, h_z, zLength * sizeof(Array), cudaMemcpyHostToDevice);

        // d_a
        Array* h_a;
        Array *d_a;
        int aLength = length + 1;
        h_a = (Array*)malloc(aLength * sizeof(Array));
        h_a[0].length = input.length;
        cudaMalloc(&h_a[0].elements, input.length * sizeof(float));
        cudaMemcpy(h_a[0].elements, input.elements, input.length * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 1; i < aLength; i++)
        {
            h_a[i].length = biases[i - 1].length;
            cudaMalloc(&h_a[i].elements, h_a[i].length * sizeof(float));
        }
        cudaMalloc(&d_a, aLength * sizeof(Array));
        cudaMemcpy(d_a, h_a, aLength * sizeof(Array), cudaMemcpyHostToDevice);
        
        // Forward Propagate
        for (size_t i = 0; i < length; i++) {
            dim3 dimBlock(biases[i].length);
            dim3 dimGrid(1);
            forwardPropagateKernel <<< dimGrid, dimBlock >>> (d_weights, d_biases, d_z, d_a, i);
            //int GRID_SIZE = (int)ceil(biases[i].length / BLOCK_SIZE);
            //forwardPropagateKernel <<< GRID_SIZE, BLOCK_SIZE >>> (d_weights, d_biases, d_z, d_a, i); //<-- something wrong
            cudaThreadSynchronize();
        }

        // d_d
        Array* h_d, * d_d;
        int dLength = length;
        h_d = (Array*)malloc(dLength * sizeof(Array));
        for (int i = 0; i < dLength; i++)
        {
            h_d[i].length = biases[i].length;
            cudaMalloc(&h_d[i].elements, h_d[i].length * sizeof(float));
        }
        cudaMalloc(&d_d, dLength * sizeof(Array));
        cudaMemcpy(d_d, h_d, dLength * sizeof(Array), cudaMemcpyHostToDevice);

        // d_output
        Array d_output;
        d_output.length = output.length;
        cudaMalloc(&d_output.elements,output.length * sizeof(float));
        cudaMemcpy(d_output.elements, output.elements, output.length * sizeof(float), cudaMemcpyHostToDevice);

        // Last Layer
        dim3 dimBlockLL(output.length);
        dim3 dimGridLL(1);
        lastLayer <<< dimGridLL, dimBlockLL >>> (h_d[dLength - 1], h_a[aLength - 1], d_output, h_z[zLength - 1]);
        //int GRID_SIZE = (int)ceil(output.length / BLOCK_SIZE);
        //lastLayer <<< GRID_SIZE, BLOCK_SIZE >>> (h_d[dLength-1],h_a[aLength-1],d_output,h_z[zLength-1]); //<-- something wrong
        cudaThreadSynchronize();

        // Remaining layers
        for (size_t i = dLength - 1; i > 0; i--)
        {
            dim3 dimBlock(h_d[i - 1].length);
            dim3 dimGrid(1);
            remainingLayers <<< dimGrid, dimBlock >>> (d_d, d_weights, d_z, i);
            //int GRID_SIZE = (int)ceil(h_d[i - 1].length / BLOCK_SIZE);
            //remainingLayers <<< GRID_SIZE, BLOCK_SIZE >>> (d_d, d_weights, d_z, i); //<-- seems to produce ok results
            cudaThreadSynchronize();
        }

        // Return results
        
        // d_dw
        Matrix* h_dw;
        //Matrix *d_dw;
        h_dw = (Matrix*)malloc(length * sizeof(Matrix));
        for (int i = 0; i < length; i++)
        {
            h_dw[i].width = dw[i].width;
            h_dw[i].height = dw[i].height;
            cudaMalloc(&h_dw[i].elements, h_dw[i].width * h_dw[i].height * sizeof(float));
        }
        //cudaMalloc(&d_dw, length * sizeof(Matrix));
        //cudaMemcpy(d_dw, h_dw, length * sizeof(Matrix), cudaMemcpyHostToDevice);

        for (size_t i = 0; i < length; i++)
        {
            assert(h_dw[i].height == h_d[i].length);
            assert(h_dw[i].width == h_a[i].length);
            int GRID_SIZE_x = (int)ceil(dw[i].width / BLOCK_SIZE);
            int GRID_SIZE_y = (int)ceil(dw[i].height / BLOCK_SIZE);
            dim3 grid(GRID_SIZE_x, GRID_SIZE_y);
            dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
            formWeightsUpdate <<< grid, threads >>> (h_dw[i],h_d[i],h_a[i]);
        }
        cudaThreadSynchronize();

        // Write results
        for (size_t i = 0; i < length; i++)
        {
            cudaMemcpy(dw[i].elements, h_dw[i].elements, dw[i].width * dw[i].height * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(db[i].elements, h_d[i].elements, db[i].length * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Free cuda memory on h_weights elements when kernel is done.
        for (size_t i = 0; i < length; i++)
        {
            cudaFree(h_weights[i].elements);
        }
        // Free cuda memory d_weights
        cudaFree(d_weights);
        // Free h_weights when kernel is done
        free(h_weights);

        // Free cuda memory on h_biases elements when kernel is done
        for (size_t i = 0; i < length; i++)
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
        //cudaFree(d_a);
        // Free h_a when kernel is done
        free(h_a);

        // Free cuda memory on h_d elements when kernel is done
        for (size_t i = 0; i < dLength; i++)
        {
            cudaFree(h_d[i].elements);
        }
        // Free cuda memory d_d
        cudaFree(d_d);
        // Free h_d when kernel is done
        free(h_d);

        // Free cuda memory d_output.elements
        cudaFree(d_output.elements);

        // Free cuda memory on h_dw elements when kernel is done
        for (size_t i = 0; i < length; i++)
        {
            cudaFree(h_dw[i].elements);
        }

        // Free h_dw when kernel is done
        free(h_dw);

        return 0;
    }
}