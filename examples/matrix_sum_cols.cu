#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SIZE 100
#define BLOCK_SIZE 10

// Kernel para sumar los elementos de cada columna de la matriz
__global__ void sum_columns(int *matrix, int *result)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < SIZE; i++)
    {
        sum += matrix[i * SIZE + col];
    }

    result[col] = sum;
}

int main()
{
    int matrix[SIZE][SIZE];
    int result[SIZE];

    // Inicializar la matriz con valores aleatorios entre 1 y 10
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            matrix[i][j] = rand() % 10 + 1;
        }
    }

    int *dev_matrix, *dev_result;

    // Reservar memoria en el dispositivo (GPU)
    cudaMalloc((void **)&dev_matrix, SIZE * SIZE * sizeof(int));
    cudaMalloc((void **)&dev_result, SIZE * sizeof(int));

    // Copiar la matriz del host al dispositivo
    cudaMemcpy(dev_matrix, matrix, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Definir el número de bloques y hilos por bloque
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Llamar al kernel
    sum_columns<<<dimGrid, dimBlock>>>(dev_matrix, dev_result);

    // Copiar el resultado de vuelta del dispositivo al host
    cudaMemcpy(result, dev_result, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar los resultados de la suma de cada columna
    printf("Suma de cada columna:\n");
    for (int i = 0; i < SIZE; i++)
    {
        printf("Columna %d: %d\n", i, result[i]);
    }

    // Liberar la memoria del dispositivo
    cudaFree(dev_matrix);
    cudaFree(dev_result);

    return 0;
}
