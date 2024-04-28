#include <stdio.h>

#define N 1000

__global__ void count_primes(int *count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    int isPrime = 1;
    for (int i = 2; i < idx; i++)
    {
        if (idx % i == 0)
        {
            isPrime = 0;
            break;
        }
    }

    if (idx > 1 && isPrime)
    {
        atomicAdd(count, 1);
    }
}

int main()
{
    int count = 0;
    int *d_count;

    cudaMalloc((void **)&d_count, sizeof(int));
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_primes<<<blocksPerGrid, threadsPerBlock>>>(d_count);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    printf("Number of primes between 1 and %d is: %d\n", N, count);

    return 0;
}
