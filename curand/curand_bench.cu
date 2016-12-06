#include <stdio.h>
#include <curand.h>

int main(){
  size_t n = 536870912;
  clock_t start=clock();
  int i;
  curandGenerator_t gen;
  float *devData;
  float f;
  cudaMalloc((void **)&devData, n * sizeof(float));
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  for(i=0;i<100;i++) curandGenerateUniform(gen, devData, n);
  printf("time1 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  cudaMemcpy(&f, devData, sizeof(float),cudaMemcpyDeviceToHost);
  printf("time2 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  curandDestroyGenerator(gen);
  cudaFree(devData);
  printf("time3 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  return 0;
}
