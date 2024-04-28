#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

// CUDA kernel to sum elements of a vector
__global__ void sum_vector(int *vec, int *result, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Perform the summation in parallel
    for (int i = tid; i < n; i += stride)
    {
        atomicAdd(result, vec[i]);
    }
}

int main()
{

    srand(time(NULL));

    // Host vectors and result
    int *h_vec, *h_result;
    h_vec = (int *)malloc(N * sizeof(int));
    h_result = (int *)malloc(sizeof(int));

    // Initialize vector elements
    for (int i = 0; i < N; ++i)
    {
        h_vec[i] = rand() % 11 + 5;
    }

    // Allocate device memory
    int *d_vec, *d_result;
    cudaMalloc((void **)&d_vec, N * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_vec, h_vec, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize result on device to zero
    cudaMemset(d_result, 0, sizeof(int));

    // Launch the kernel
    sum_vector<<<16, 16>>>(d_vec, d_result, N);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sum
    printf("Sum of vector elements: %d\n", *h_result);

    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_result);

    // Free host memory
    free(h_vec);
    free(h_result);

    return 0;
}
