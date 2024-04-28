#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100

// CUDA kernel to sum two matrices
__global__ void sum_matrices(int *mat1, int *mat2, int *result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        *(result + row * N + col) = *(mat1 + row * N + col) + *(mat2 + row * N + col);
    }
}

// Function to initialize a matrix with random values between 10 and 20
void initialize_matrix(int *mat, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            mat[i * size + j] = rand() % 11 + 10; // Values between 10 and 20
        }
    }
}

// Function to print a matrix
void print_matrix(int *mat, int size)
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
    srand(time(NULL));

    // Host matrices and result
    int *h_mat1, *h_mat2, *h_result;

    h_mat1 = (int *)malloc(N * N * sizeof(int));
    h_mat2 = (int *)malloc(N * N * sizeof(int));
    h_result = (int *)malloc(N * N * sizeof(int));

    // Initialize matrices with random values
    initialize_matrix(h_mat1, N);
    initialize_matrix(h_mat2, N);

    // Print matrices if needed
    // printf("Matrix 1:\n");
    // print_matrix(h_mat1, N);
    // printf("\nMatrix 2:\n");
    // print_matrix(h_mat2, N);

    // Device matrices and result
    int *d_mat1, *d_mat2, *d_result;
    cudaMalloc((void **)&d_mat1, N * N * sizeof(int));
    cudaMalloc((void **)&d_mat2, N * N * sizeof(int));
    cudaMalloc((void **)&d_result, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_mat1, h_mat1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16); // 16x16 thread block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    sum_matrices<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_result);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result if needed
    // printf("\nResult Matrix:\n");
    // print_matrix(h_result, N);

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
