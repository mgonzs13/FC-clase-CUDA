#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

// CUDA kernel to calculate mean of a vector
__global__ void calculate_mean(int *vec, float *result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    int count = 0;

    // Perform the summation in parallel
    for (int i = tid; i < N; i += stride)
    {
        sum += vec[i];
        count++;
    }

    atomicAdd(result, sum);
    atomicAdd(result + 1, count);
}

int main()
{
    srand(time(NULL));

    // Vector size
    size_t bytes = N * sizeof(int);

    // Host vector and result
    int *h_vec;
    h_vec = (int *)malloc(bytes);

    // Initialize vector elements
    for (int i = 0; i < N; ++i)
    {
        h_vec[i] = rand() % 100; // Example: fill vector with random integers
    }

    // Allocate device memory
    int *d_vec;
    float *d_result;
    cudaMalloc((void **)&d_vec, bytes);
    cudaMalloc((void **)&d_result, 2 * sizeof(float)); // 2 floats: sum, count

    // Copy data from host to device
    cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, 2 * sizeof(float)); // Initialize result on device to zero

    // Define block size and grid size
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // Launch the kernel
    calculate_mean<<<grid_size, block_size>>>(d_vec, d_result);

    // Copy the partial sum and count back to the host
    float h_partialResult[2] = {0};
    cudaMemcpy(h_partialResult, d_result, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate final mean on the host
    float mean = h_partialResult[0] / h_partialResult[1];

    // Print the mean
    printf("Mean of vector elements: %.2f\n", mean);

    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_result);

    // Free host memory
    free(h_vec);

    return 0;
}
