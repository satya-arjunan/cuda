#include <stdio.h>
#include <driver_types.h>
#include <cuda_runtime.h>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 1

int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void kernel0(float *d_a, float *d_b, const unsigned int M, const unsigned int N)
{
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((tidx < M)&&(tidy < N)) {

        d_b[tidy * M + tidx] = d_a[tidy * M + tidx];

    }

}

int main()
{
    const unsigned int M = 32*1000;
    const unsigned int N = 1*1000;

    float *d_a; cudaMalloc((void**)&d_a, M*N*sizeof(float));
    float *d_b; cudaMalloc((void**)&d_b, M*N*sizeof(float));

    dim3 dimGrid(iDivUp(M, BLOCKSIZE_X), iDivUp(N, BLOCKSIZE_Y));
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    printf("grid x:%d y:%d z:%d, block x:%d y:%d z:%d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    kernel0<<<dimGrid, dimBlock>>>(d_a, d_b, M, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaDeviceReset();
    return 0;
}
