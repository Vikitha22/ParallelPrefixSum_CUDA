
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>

using namespace std;

#define N 8

// Blelloch Work-Efficient Scan
__global__ void blellochScan(int *input, int *output)
{
    __shared__ int temp[N];

    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    // UP-SWEEP PHASE
    for (int stride = 1; stride < N; stride *= 2)
    {
        int index = (tid + 1) * stride * 2 - 1;

        if (index < N)
            temp[index] += temp[index - stride];

        __syncthreads();
    }

    // Set last element to zero
    if (tid == 0)
        temp[N - 1] = 0;

    __syncthreads();

    // DOWN-SWEEP PHASE
    for (int stride = N / 2; stride >= 1; stride /= 2)
    {
        int index = (tid + 1) * stride * 2 - 1;

        if (index < N)
        {
            int value = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += value;
        }

        __syncthreads();
    }

    // Exclusive scan output
    output[tid] = temp[tid];
}

int main()
{
    int h_input[N] = {3, 1, 7, 0, 4, 1, 6, 3};
    int h_output[N] = {0};
    int h_reference[N] = {0};

    // CPU Exclusive Scan
    partial_sum(h_input, h_input + N - 1, h_reference + 1);

    int *d_input, *d_output;

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int),
               cudaMemcpyHostToDevice);

    blellochScan<<<1, N>>>(d_input, d_output);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        cout << "Kernel launch failed: "
             << cudaGetErrorString(err) << endl;
        return 1;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        cout << "Kernel execution failed: "
             << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaMemcpy(h_output, d_output, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    cout << "\nBLELLOCH WORK-EFFICIENT SCAN\n";

    cout << "\nInput: ";
    for (int i = 0; i < N; i++)
        cout << h_input[i] << " ";

    cout << "\n\nCPU Exclusive Scan: ";
    for (int i = 0; i < N; i++)
        cout << h_reference[i] << " ";

    cout << "\n\nGPU Exclusive Scan: ";
    for (int i = 0; i < N; i++)
        cout << h_output[i] << " ";

    bool correct = true;

    for (int i = 0; i < N; i++)
        if (h_output[i] != h_reference[i])
            correct = false;

    cout << "\n\nVerification: "
         << (correct ? "PASSED" : "FAILED") << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return correct ? 0 : 1;
}
