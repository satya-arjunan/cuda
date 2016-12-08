#include <stdio.h>
#include <curand.h>

//best results: 41.3 BUPS

int main(){
  size_t n = 536870912;
  int i;
  curandGenerator_t gen;
  float *devData;
  float f;
  cudaMalloc((void **)&devData, n * sizeof(float));
  //curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MT19937);
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  clock_t start=clock();
  for(i=0;i<100;i++) curandGenerateUniform(gen, devData, n);
  cudaDeviceSynchronize();
  printf("time1 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  cudaMemcpy(&f, devData, sizeof(float),cudaMemcpyDeviceToHost);
  printf("time2 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  curandDestroyGenerator(gen);
  cudaFree(devData);
  printf("time3 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  return 0;
}
