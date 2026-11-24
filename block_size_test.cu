
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;

#define N (1 << 16)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#define NUM_BLOCKS (N / BLOCK_SIZE)
#define ITERATIONS 20

// Scan each input block
__global__ void blockScan(int *input, int *output, int *blockSums)
{
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * BLOCK_SIZE + tid;

    temp[tid] = input[index];
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2)
    {
        int pos = (tid + 1) * stride * 2 - 1;

        if (pos < BLOCK_SIZE)
            temp[pos] += temp[pos - stride];

        __syncthreads();
    }

    if (tid == 0)
    {
        blockSums[blockIdx.x] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0;
    }

    __syncthreads();

    // Down-sweep
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
    {
        int pos = (tid + 1) * stride * 2 - 1;

        if (pos < BLOCK_SIZE)
        {
            int value = temp[pos - stride];
            temp[pos - stride] = temp[pos];
            temp[pos] += value;
        }

        __syncthreads();
    }

    output[index] = temp[tid];
}

// Scan the block sums
__global__ void scanBlockSums(int *blockSums, int *offsets)
{
    __shared__ int temp[NUM_BLOCKS];

    int tid = threadIdx.x;

    temp[tid] = blockSums[tid];
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < NUM_BLOCKS; stride *= 2)
    {
        int pos = (tid + 1) * stride * 2 - 1;

        if (pos < NUM_BLOCKS)
            temp[pos] += temp[pos - stride];

        __syncthreads();
    }

    if (tid == 0)
        temp[NUM_BLOCKS - 1] = 0;

    __syncthreads();

    // Down-sweep
    for (int stride = NUM_BLOCKS / 2; stride >= 1; stride /= 2)
    {
        int pos = (tid + 1) * stride * 2 - 1;

        if (pos < NUM_BLOCKS)
        {
            int value = temp[pos - stride];
            temp[pos - stride] = temp[pos];
            temp[pos] += value;
        }

        __syncthreads();
    }

    offsets[tid] = temp[tid];
}

// Add block offsets
__global__ void addBlockOffsets(int *output, int *offsets)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    output[index] += offsets[blockIdx.x];
}

// Execute the complete GPU scan
void runGpuScan(int *d_input, int *d_output,
                int *d_blockSums, int *d_offsets)
{
    blockScan<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_input, d_output, d_blockSums);

    scanBlockSums<<<1, NUM_BLOCKS>>>(
        d_blockSums, d_offsets);

    addBlockOffsets<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_output, d_offsets);
}

int main()
{
    vector<int> h_input(N);
    vector<int> h_output(N);
    vector<int> h_reference(N, 0);

    // Initialize input
    for (int i = 0; i < N; i++)
        h_input[i] = (i % 5) + 1;

    // CPU Exclusive Scan
    auto cpuStart = chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; i++)
    {
        h_reference[0] = 0;

        partial_sum(h_input.begin(), h_input.end() - 1,
                    h_reference.begin() + 1);
    }

    auto cpuEnd = chrono::high_resolution_clock::now();

    double cpuTime =
        chrono::duration<double, milli>(cpuEnd - cpuStart).count()
        / ITERATIONS;

    int *d_input, *d_output, *d_blockSums, *d_offsets;

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    cudaMalloc((void**)&d_blockSums, NUM_BLOCKS * sizeof(int));
    cudaMalloc((void**)&d_offsets, NUM_BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
               cudaMemcpyHostToDevice);

    // Warm-up run
    runGpuScan(d_input, d_output, d_blockSums, d_offsets);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        cout << "Warm-up error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // CUDA event timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < ITERATIONS; i++)
    {
        runGpuScan(d_input, d_output, d_blockSums, d_offsets);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalGpuTime = 0.0f;
    cudaEventElapsedTime(&totalGpuTime, start, stop);

    double gpuTime = totalGpuTime / ITERATIONS;

    // Check kernel execution
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        cout << "Kernel error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Copy GPU result to CPU
    cudaMemcpy(h_output.data(), d_output, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Verify result
    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        if (h_output[i] != h_reference[i])
        {
            correct = false;
            break;
        }
    }

    // Display results
    cout << fixed << setprecision(6);

    cout << "\nMULTI-BLOCK BLELLOCH SCAN WITH PERFORMANCE TIMING\n";

    cout << "\nArray size: " << N;
    cout << "\nBlock size: " << BLOCK_SIZE;
    cout << "\nNumber of blocks: " << NUM_BLOCKS;
    cout << "\nTiming repetitions: " << ITERATIONS;

    cout << "\n\nFirst 10 input elements: ";
    for (int i = 0; i < 10; i++)
        cout << h_input[i] << " ";

    cout << "\n\nFirst 10 GPU output elements: ";
    for (int i = 0; i < 10; i++)
        cout << h_output[i] << " ";

    cout << "\n\nLast GPU output element: "
         << h_output[N - 1];

    cout << "\n\nCPU average time: "
         << cpuTime << " ms";

    cout << "\nGPU average time: "
         << gpuTime << " ms";

    cout << "\nGPU speedup: "
         << cpuTime / gpuTime << "x";

    cout << "\n\nVerification: "
         << (correct ? "PASSED" : "FAILED") << endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    cudaFree(d_offsets);

    return correct ? 0 : 1;
}
