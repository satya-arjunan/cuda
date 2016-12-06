#include<stdio.h>
#include<mt19937.h>

int main(void){
  size_t n = 536870912;
  unsigned int* dev_out;
  cudaMalloc((void**)&dev_out,n*sizeof(unsigned int));
  mt19937_state state; 
  mt19937_init_device_consts_();
  mt19937_init_(&state); 
  clock_t start=clock();
  mt19937_generate_gpu_array_(&state,dev_out,n);
  cudaDeviceSynchronize();
  printf("time1 = %f seconds\n",(float)(clock()-start)/CLOCKS_PER_SEC);
  mt19937_free_device_consts_();
  cudaFree(dev_out);
  return 0;
}

