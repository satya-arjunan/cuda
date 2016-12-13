#include <stdio.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

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

typedef int32_t mol_t;
typedef uint32_t umol_t;
typedef int32_t mol2_t;
typedef uint32_t umol2_t;

__global__
void concurrent_walk(
    const unsigned M,
    const unsigned N,
    umol_t* res, 
    umol_t* mols_) {

  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  if((tidx < M)&&(tidy < N)) { 
    mols_[tidy * M + tidx] = mols_[tidy * M + tidx]+threadIdx.x;
  }
}

__global__
void concurrent_walk2(
    const unsigned M,
    umol_t* res, 
    umol_t* mols_) {

  const unsigned block_jobs(M/gridDim.x);
  unsigned index(blockIdx.x*block_jobs);
  unsigned end_index(index+block_jobs);
  while(index < end_index) {
    res[index] = mols_[index]+threadIdx.x;
    index += blockDim.x;
  }
}
/*
int main() {
  const unsigned size(53673000);
  const unsigned int M = 53673;
  const unsigned int N = 1000;
  umol_t* mols_;
  umol_t* res;
  cudaMalloc((void**)&mols_, size*sizeof(umol_t));
  cudaMalloc((void**)&res, size*sizeof(umol_t));
  dim3 dimGrid(iDivUp(M, BLOCKSIZE_X), iDivUp(N, BLOCKSIZE_Y));
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
  //printf("grid x:%d y:%d z:%d, block x:%d y:%d z:%d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  //concurrent_walk<<<dimGrid, dimBlock>>>(M, mols_, res);

  clock_t start=clock();
  for(unsigned i(0); i != 1000; ++i) {
    concurrent_walk2<<<64, 256>>>(size, mols_, res);
  }
  cudaDeviceSynchronize();
  printf("time1 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
*/
int main() {
  const unsigned size(53673000);
  const unsigned int M = 536730;
  const unsigned int N = 1;
  umol_t* mols_;
  umol_t* res;
  cudaMalloc((void**)&mols_, M*N*sizeof(umol_t));
  cudaMalloc((void**)&res, M*N*sizeof(umol_t));
  dim3 dimGrid(iDivUp(M, BLOCKSIZE_X), iDivUp(N, BLOCKSIZE_Y));
  dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
  //printf("grid x:%d y:%d z:%d, block x:%d y:%d z:%d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  clock_t start=clock();
  for(unsigned i(0); i != 100000; ++i) {
    concurrent_walk<<<dimGrid, dimBlock>>>(M, N, mols_, res);
  }
  cudaDeviceSynchronize();
  printf("time1 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
