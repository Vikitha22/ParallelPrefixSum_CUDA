
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

using namespace std;

#define N 8

// Hillis-Steele Parallel Prefix Sum
__global__ void hillisSteeleScan(int *input, int *output)
{
    __shared__ int temp[N];
    __shared__ int buffer[N];

    int tid = threadIdx.x;

    // Load input into shared memory
    temp[tid] = input[tid];
    __syncthreads();

    // Hillis-Steele scan
    for (int offset = 1; offset < N; offset *= 2)
    {
        buffer[tid] = temp[tid];
        __syncthreads();

        if (tid >= offset)
            buffer[tid] += temp[tid - offset];

        __syncthreads();

        temp[tid] = buffer[tid];
        __syncthreads();
    }

    output[tid] = temp[tid];
}

int main()
{
    int h_input[N] = {3, 1, 7, 0, 4, 1, 6, 3};

    int h_output[N] = {0};
    int h_reference[N] = {0};

    // CPU Inclusive Scan
    partial_sum(h_input, h_input + N, h_reference);

    int *d_input = nullptr;
    int *d_output = nullptr;

    // Allocate GPU memory
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, N * sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Input allocation failed: "
             << cudaGetErrorString(err) << endl;
        return 1;
    }

    err = cudaMalloc((void**)&d_output, N * sizeof(int));
    if (err != cudaSuccess)
    {
        cout << "Output allocation failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        return 1;
    }

    // Copy input to GPU
    err = cudaMemcpy(d_input, h_input, N * sizeof(int),
                     cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        cout << "Host-to-device copy failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Launch CUDA kernel
    hillisSteeleScan<<<1, N>>>(d_input, d_output);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        cout << "Kernel launch failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        cout << "Kernel execution failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Copy result from GPU
    err = cudaMemcpy(h_output, d_output, N * sizeof(int),
                     cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        cout << "Device-to-host copy failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Display results
    cout << "\nPARALLEL PREFIX SUM - HILLIS STEELE\n";

    cout << "\nInput: ";
    for (int i = 0; i < N; i++)
        cout << h_input[i] << " ";

    cout << "\n\nCPU Inclusive Scan: ";
    for (int i = 0; i < N; i++)
        cout << h_reference[i] << " ";

    cout << "\n\nGPU Inclusive Scan: ";
    for (int i = 0; i < N; i++)
        cout << h_output[i] << " ";

    // Verify results
    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        if (h_output[i] != h_reference[i])
            correct = false;
    }

    cout << "\n\nVerification: "
         << (correct ? "PASSED" : "FAILED") << endl;

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return correct ? 0 : 1;
}
