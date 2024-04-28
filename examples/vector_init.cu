#include <stdio.h>

#define N 128

__global__ void initVector(int *vector)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        *(vector + i) = index;
    }
}

int main()
{

    int *host_vector;
    int *device_vector;

    // Reserva de memoria en el host
    host_vector = (int *)malloc(N * sizeof(int));

    // Reserva de memoria en el device (GPU)
    cudaMalloc((void **)&device_vector, N * sizeof(int));

    // Llama al kernel con la configuración de tamaño de bloque y número de bloques
    initVector<<<8, 8>>>(device_vector);

    // Copia los datos de la GPU al host
    cudaMemcpy(host_vector, device_vector, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprime el vector inicializado
    printf("Vector inicializado:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", host_vector[i]);
    }
    printf("\n");

    // Liberar memoria
    free(host_vector);
    cudaFree(device_vector);

    return 0;
}
