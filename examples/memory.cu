#include <stdio.h>

#define N 10

__global__ void sum_to_vector(float *A, float *B)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        B[index] = A[index] + N;
    }
}

int main()
{
    float h_A[N];
    float h_B[N];

    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)(i + 1);
    }

    float *d_A;
    float *d_B;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    sum_to_vector<<<gridSize, blockSize>>>(d_A, d_B);

    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant Vector:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%.2f\t", h_B[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
