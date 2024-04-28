#include <stdio.h>

#define N 1024

// CUDA kernel to sum elements of a vector
__global__ void sum_vector(int *vec, int *result)
{
    __shared__ int partialSum[256]; // Shared memory for each block

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    partialSum[tid] = 0;

    while (i < N)
    {
        partialSum[tid] += vec[i];
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, partialSum[0]);
    }
}

int main()
{
    int *h_vec, *d_vec, *h_result, *d_result;
    size_t bytes = N * sizeof(int);
    h_vec = (int *)malloc(bytes);
    h_result = (int *)malloc(sizeof(int));

    int sum = 0;
    for (int i = 0; i < N; ++i)
    {
        h_vec[i] = i + 1;
        sum += i + 1;
    }
    printf("%d\n", sum);

    cudaMalloc((void **)&d_vec, bytes);
    cudaMalloc((void **)&d_result, sizeof(int));

    cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    sum_vector<<<grid_size, block_size>>>(d_vec, d_result);

    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of vector elements: %d\n", *h_result);

    cudaFree(d_vec);
    cudaFree(d_result);
    free(h_vec);
    free(h_result);

    return 0;
}
