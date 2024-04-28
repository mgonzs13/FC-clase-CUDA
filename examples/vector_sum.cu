#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

__global__ void sum_vectors(float *A, float *B, float *C)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        C[index] = A[index] + B[index];
    }
}

int main()
{
    float h_A[N], h_B[N], h_C[N];

    for (int i = 0; i < N; ++i)
    {
        h_A[i] = rand() % 10 + 1;
        h_B[i] = rand() % 10 + 1;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    sum_vectors<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant Vector:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%.2f + %.2f = %.2f\t", h_A[i], h_B[i], h_C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
