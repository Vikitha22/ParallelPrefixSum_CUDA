
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;

#define N (1 << 20)
#define BLOCK_SIZE 1024
#define NUM_BLOCKS (N / BLOCK_SIZE)
#define ITERATIONS 20

// Hillis-Steele scan within each block
__global__ void blockHillisScan(int *input, int *output, int *blockSums)
{
    __shared__ int temp[BLOCK_SIZE];
    __shared__ int buffer[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * BLOCK_SIZE + tid;

    temp[tid] = input[index];
    __syncthreads();

    // Inclusive Hillis-Steele scan
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2)
    {
        buffer[tid] = temp[tid];
        __syncthreads();

        if (tid >= offset)
            buffer[tid] += temp[tid - offset];

        __syncthreads();

        temp[tid] = buffer[tid];
        __syncthreads();
    }

    // Convert inclusive result to exclusive result
    output[index] = temp[tid] - input[index];

    // Store total of this block
    if (tid == BLOCK_SIZE - 1)
        blockSums[blockIdx.x] = temp[tid];
}

// Hillis-Steele scan of block sums
__global__ void scanBlockSums(int *blockSums, int *scannedSums)
{
    __shared__ int temp[NUM_BLOCKS];
    __shared__ int buffer[NUM_BLOCKS];

    int tid = threadIdx.x;

    temp[tid] = blockSums[tid];
    __syncthreads();

    for (int offset = 1; offset < NUM_BLOCKS; offset *= 2)
    {
        buffer[tid] = temp[tid];
        __syncthreads();

        if (tid >= offset)
            buffer[tid] += temp[tid - offset];

        __syncthreads();

        temp[tid] = buffer[tid];
        __syncthreads();
    }

    scannedSums[tid] = temp[tid];
}

// Add preceding block totals
__global__ void addBlockOffsets(int *output, int *scannedSums)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (blockIdx.x > 0)
        output[index] += scannedSums[blockIdx.x - 1];
}

// Execute complete GPU scan
void runGpuScan(int *d_input, int *d_output,
                int *d_blockSums, int *d_scannedSums)
{
    blockHillisScan<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_input, d_output, d_blockSums);

    scanBlockSums<<<1, NUM_BLOCKS>>>(
        d_blockSums, d_scannedSums);

    addBlockOffsets<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_output, d_scannedSums);
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

    int *d_input, *d_output, *d_blockSums, *d_scannedSums;

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    cudaMalloc((void**)&d_blockSums, NUM_BLOCKS * sizeof(int));
    cudaMalloc((void**)&d_scannedSums, NUM_BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
               cudaMemcpyHostToDevice);

    // Warm-up run
    runGpuScan(d_input, d_output, d_blockSums, d_scannedSums);

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
        runGpuScan(d_input, d_output, d_blockSums, d_scannedSums);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalGpuTime = 0.0f;
    cudaEventElapsedTime(&totalGpuTime, start, stop);

    double gpuTime = totalGpuTime / ITERATIONS;

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

    cout << "\nMULTI-BLOCK HILLIS-STEEL EXCLUSIVE SCAN\n";

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
    cudaFree(d_scannedSums);

    return correct ? 0 : 1;
}
