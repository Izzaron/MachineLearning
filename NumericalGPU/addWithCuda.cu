#include "NumericalGPU.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// This function is to be removed. It is merely kept as an example of error handling

namespace NumericalGPU {
    __global__ void addKernel(float* c, const float* a, const float* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] - b[i];
    }

    cudaError_t addOnDevice(float* c, const float* a, const float* b, float* dev_c, float* dev_a, float* dev_b, unsigned int size) {

        cudaError_t cudaStatus;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return cudaStatus;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        // Launch a kernel on the GPU with one thread for each element.
        addKernel << <1, size >> > (dev_c, dev_a, dev_b);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            return cudaStatus;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
    }

    // Helper function for using CUDA to add vectors in parallel.
    int addWithCuda(float* c, const float* a, const float* b, unsigned int size)
    {
        float* dev_a = 0;
        float* dev_b = 0;
        float* dev_c = 0;

        //Add vectors in parallel.
        cudaError_t cudaStatus = addOnDevice(c, a, b, dev_c, dev_a, dev_b, size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        return 0;
    }
}
