#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

// CUDA kernel to multiply two matrices
__global__ void multiply_matrices(int *mat1, int *mat2, int *result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {

        int sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += mat1[row * N + i] * mat2[i * N + col];
        }
        result[row * N + col] = sum;
    }
}

// Function to initialize a matrix with random values between 5 and 15
void initializeMatrix(int *mat, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            mat[i * size + j] = rand() % 11 + 5; // Values between 5 and 15
        }
    }
}

// Function to print a matrix
void printMatrix(int *mat, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%d\t", mat[i * size + j]);
        }
        printf("\n");
    }
}

int main()
{

    // Host matrices and result
    int *h_mat1, *h_mat2, *h_result;
    size_t bytes = N * N * sizeof(int);

    h_mat1 = (int *)malloc(bytes);
    h_mat2 = (int *)malloc(bytes);
    h_result = (int *)malloc(bytes);

    // Initialize matrices with random values
    initializeMatrix(h_mat1, N);
    initializeMatrix(h_mat2, N);

    // Print matrices if needed
    // printf("Matrix 1:\n");
    // printMatrix(h_mat1, N);
    // printf("\nMatrix 2:\n");
    // printMatrix(h_mat2, N);

    // Device matrices and result
    int *d_mat1, *d_mat2, *d_result;
    cudaMalloc((void **)&d_mat1, bytes);
    cudaMalloc((void **)&d_mat2, bytes);
    cudaMalloc((void **)&d_result, bytes);

    // Copy data from host to device
    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, bytes, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(N, N); // 16x16 thread block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    multiply_matrices<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_result);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    // Print the result if needed
    printf("\nResult Matrix:\n");
    printMatrix(h_result, N);

    // Free device memory
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    // Free host memory
    free(h_mat1);
    free(h_mat2);
    free(h_result);

    return 0;
}
