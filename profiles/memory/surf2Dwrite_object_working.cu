//to compile with global memory: nvcc -O3 -gencode arch=compute_52,code=sm_52 -DUSE_GLOBAL surf2Dwrite_ex.cu
//to compile with surface memory: nvcc -O3 -gencode arch=compute_52,code=sm_52 surf2Dwrite_ex.cu

#include <stdio.h>
#include <iostream>

typedef float mytype;
const int blk_dim=16;

#define my_N 1000
#define A_VAL 1
#define B_VAL 2

void CUDA_SAFE_CALL(cudaError_t call, int line) {
    switch (call) {
    case cudaSuccess:
        break;
    default:
        printf("ERROR at line :%i.%d' ' %s\n",
            line, call, cudaGetErrorString(call));
        exit(-1);
        break;
    }

}

#ifdef USE_GLOBAL
__global__ void mul(const mytype * __restrict__ d_a, const mytype * __restrict__ d_b, mytype * __restrict__ d_c, const int N)
#else
__global__ void mul(const int N, cudaSurfaceObject_t a_surf, cudaSurfaceObject_t b_surf, cudaSurfaceObject_t c_surf)
#endif
{
    mytype a, b, c, temp;
    int i;

    unsigned int x = blockIdx.x * blockDim.x + (threadIdx.x);
    unsigned int y = blockIdx.y * blockDim.y + (threadIdx.y);
    if (x < N && y < N) {

        temp = 0;
        for (i = 0; i < N; i++) {
#ifdef USE_GLOBAL
            a = d_a[x*N+i];
            b = d_b[i*N+y];
#else
            surf2Dread( & a, a_surf, (x) * sizeof(mytype), i);
            surf2Dread( & b, b_surf, (i) * sizeof(mytype), y);
#endif
            temp += a * b;
        }
        c = temp;
#ifdef USE_GLOBAL
        d_c[x*N+y] = c;
#else
        // Write to output surface
        surf2Dwrite(c, c_surf, x * sizeof(mytype), y);
#endif
    }
}

int main() {
    cudaSurfaceObject_t a_surf;
    cudaSurfaceObject_t b_surf;
    cudaSurfaceObject_t c_surf;
    const int N = my_N;
    mytype *a, *b, *c, *d_a, *d_b, *d_c;
    int i, j;
    clock_t t1, t2;
    cudaArray * da, * db, * dc;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < mytype > ();

    dim3 dimBlock(blk_dim, blk_dim);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
    int s = N * N * sizeof(mytype);

    a = (mytype *)malloc(s);
    b = (mytype *)malloc(s);
    c = (mytype *)malloc(s);

    CUDA_SAFE_CALL(cudaMalloc(&d_a, s), __LINE__);
    CUDA_SAFE_CALL(cudaMalloc(&d_b, s), __LINE__);
    CUDA_SAFE_CALL(cudaMalloc(&d_c, s), __LINE__);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            a[i*N+j] = A_VAL;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            b[i*N+j] = B_VAL;

    CUDA_SAFE_CALL(cudaMallocArray( & da, & channelDesc, N, N, cudaArraySurfaceLoadStore), __LINE__);
    CUDA_SAFE_CALL(cudaMallocArray( & db, & channelDesc, N, N, cudaArraySurfaceLoadStore), __LINE__);
    CUDA_SAFE_CALL(cudaMallocArray( & dc, & channelDesc, N, N, cudaArraySurfaceLoadStore), __LINE__);


    CUDA_SAFE_CALL(cudaMemcpyToArray(da, 0, 0, a, s, cudaMemcpyHostToDevice), __LINE__);
    CUDA_SAFE_CALL(cudaMemcpyToArray(db, 0, 0, b, s, cudaMemcpyHostToDevice), __LINE__);


    cudaResourceDesc a_res;
    memset(&a_res, 0, sizeof(cudaResourceDesc));
    a_res.resType = cudaResourceTypeArray;
    a_res.res.array.array = da;
    cudaCreateSurfaceObject(&a_surf, &a_res);


    cudaResourceDesc b_res;
    memset(&b_res, 0, sizeof(cudaResourceDesc));
    b_res.resType = cudaResourceTypeArray;
    b_res.res.array.array = db;
    cudaCreateSurfaceObject(&b_surf, &b_res);


    cudaResourceDesc c_res;
    memset(&c_res, 0, sizeof(cudaResourceDesc));
    c_res.resType = cudaResourceTypeArray;
    c_res.res.array.array = dc;
    cudaCreateSurfaceObject(&c_surf, &c_res);

#ifdef USE_GLOBAL
    CUDA_SAFE_CALL(cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice), __LINE__);
    CUDA_SAFE_CALL(cudaMemcpy(d_b, b, s, cudaMemcpyHostToDevice), __LINE__);
#endif
    t1 = clock();
#ifdef USE_GLOBAL
    mul <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, N);
#else
    mul <<<dimGrid, dimBlock>>> (N, a_surf, b_surf, c_surf);
#endif
    cudaDeviceSynchronize();
    t2 = clock();

    CUDA_SAFE_CALL(cudaMemcpyFromArray(c, dc, 0, 0, s, cudaMemcpyDeviceToHost), __LINE__);
#ifdef USE_GLOBAL
    CUDA_SAFE_CALL(cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost), __LINE__);
#endif

    double t3 = (double) t2 - (double) t1;
    t3 = t3 / CLOCKS_PER_SEC;

    printf("\n CUDA time :%lf\n", t3);
    for (i=0; i < N*N; i++)
      if(c[i] != A_VAL*B_VAL*N) {std::cout << "mismatch at: " << i << ", was: " << c[i] << " should be: " << A_VAL*B_VAL*N << std::endl;  return 1;}

    CUDA_SAFE_CALL(cudaFreeArray(da), __LINE__);
    CUDA_SAFE_CALL(cudaFreeArray(db), __LINE__);
    CUDA_SAFE_CALL(cudaFreeArray(dc), __LINE__);
    std::cout << "Success!"  << std::endl;
    return 0;
}
