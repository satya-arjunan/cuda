#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

#include "utils_timer.h"

// CUDA helper functions checkCudaErrors for host & getLastCudaError for device
// bundeled the headers into this project, otherwise found in the samples from the
// cuda toolkit
#include "helper_cuda.h"

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <curand.h>
#include <curand_kernel.h>


#ifdef _WIN32
// printf() is only supported for devices of compute capability 2.0 and higher
#   if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#   define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#   endif
#endif

#ifndef NULL
#def NULL ((void *)0)
#endif

#define cudaSyncAndCheck() checkCudaErrors(cudaDeviceSynchronize())
#define CURAND_CALL(x)	curandCheck(x)


void curandCheck(curandStatus result)
{
    if(result != CURAND_STATUS_SUCCESS)
    {
        printf("Error at %s : %d\n" ,__FILE__,__LINE__);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__host__ static __inline__ float _rand()
{
    return (0.0 + (float)(rand())/((float)(RAND_MAX/(1.0f - 0.0f))));
}


struct psrngen
{
    __host__ __device__ psrngen(float _a, float _b) : a(_a), b(_b) {;}

    __host__ __device__ float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
    float a, b;

};

struct psrngen_curand
{
    __host__ __device__ psrngen_curand(float _a, float _b) : a(_a), b(_b) {;}

    __device__ float operator()(const unsigned int n) const
    {
        curandState s;
        curand_init(n, 0, 0, &s);
        return curand_uniform(&s);
    }
    float a, b;

};

struct psrngeni
{
    __host__ __device__ psrngeni(unsigned _a, unsigned _b) : a(_a), b(_b) {;}

    __host__ __device__ unsigned operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<unsigned> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
    unsigned a, b;

};


__device__ float generate(curandState* globalState, const size_t ind)
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}


__global__ void set_random_number_from_kernels(float* _ptr, curandState* globalState, const size_t _points)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < _points)
    {
        _ptr[idx] = generate(globalState, idx);
    }
}


__global__ void initialise_curand_on_kernels(curandState * state, unsigned long seed, const size_t _points)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);

}


extern "C"
void testRandomGeneration()
{
    utils::stopclock* s;
    utils::createTimer(&s);
    size_t n = 10;
    double total_device, total_host = 0;
    srand(time(NULL));

    std::cerr<<"------------------------------------------------------------------------"<<std::endl;
    std::cerr<<"------------------Random number generation speed tests------------------"<<std::endl;
    std::cerr<<"------------------------------------------------------------------------"<<std::endl;
    std::cerr<<std::endl<<"Beginning thrust Host random generation using generate..."<<std::endl;

    for (size_t t =0; t < 2; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            s->startTimer();
            thrust::host_vector<float> h_vec(n, 0);
            thrust::generate(h_vec.begin(), h_vec.end(), _rand);
            //thrust::device_vector<float> d_vec(h_vec.begin(), h_vec.end());
            s->stopTimer();

            total_host+=s->getTimeInMilliseconds();
            h_vec.clear();
            thrust::host_vector<float>().swap(h_vec);

            //d_vec.clear();
            //thrust::device_vector<float>().swap(d_vec);
        }

        std::cerr<<"For n = "<<n<<", the CPU took, on average: "<<total_host/static_cast<double>(100)<<std::endl;

        total_host = 0;
        n*=1000;
    }

    std::cerr<<std::endl<<"Beginning thrust Device random generation using counting itr + transform..."<<std::endl;
    n= 10;
    total_device = 0;

    for (size_t t =0; t < 3; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            s->startTimer();
            int _seed = rand();
            thrust::device_vector<float> d_data(n);
            thrust::counting_iterator<unsigned int> index_sequence_begin(_seed);
            thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + (n), d_data.begin(), psrngen(0.0f, 1.0f));
            s->stopTimer();

            total_device+=s->getTimeInMilliseconds();

            d_data.clear();
            thrust::device_vector<float>().swap(d_data);
        }

        std::cerr<<"For n = "<<n<<", the GPU took, on average: "<<total_device/static_cast<double>(100)<<std::endl;
        total_device = 0;
        n*=1000;
    }

    std::cerr<<std::endl<<"Beginning thrust-curand Device random generation using counting itr + transform..."<<std::endl;
    n= 10;
    total_device = 0;

    for (size_t t =0; t < 3; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            s->startTimer();
            int _seed = rand();
            thrust::device_vector<float> d_data(n);
            thrust::counting_iterator<unsigned int> index_sequence_begin(_seed);
            thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + (n), d_data.begin(), psrngen_curand(0.0f, 1.0f));
            s->stopTimer();

            total_device+=s->getTimeInMilliseconds();

            d_data.clear();
            thrust::device_vector<float>().swap(d_data);
        }

        std::cerr<<"For n = "<<n<<", the GPU took, on average: "<<total_device/static_cast<double>(100)<<std::endl;
        total_device = 0;
        n*=1000;
    }

    std::cerr<<std::endl<<"Beginning thrust Device unsigned random generation using counting itr + transform..."<<std::endl;
    n= 10;
    total_device = 0;

    for (size_t t =0; t < 3; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            s->startTimer();
            int _seed = rand();
            thrust::device_vector<unsigned> d_data(n);
            thrust::counting_iterator<unsigned int> index_sequence_begin(_seed);
            thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + (n), d_data.begin(), psrngeni(0, 12));
            s->stopTimer();

            total_device+=s->getTimeInMilliseconds();

            d_data.clear();
            thrust::device_vector<unsigned>().swap(d_data);
        }

        std::cerr<<"For n = "<<n<<", the GPU took, on average: "<<total_device/static_cast<double>(100)<<std::endl;
        total_device = 0;
        n*=1000;
    }

    std::cerr<<std::endl<<"Beginning cuRand (Host API), device random generation..."<<std::endl;
    n = 10;
    total_device = 0;

    for (size_t t =0; t < 3; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            s->startTimer();
            float *deviceData/*, *hostData*/;
            curandGenerator_t gen;
            int _seed = rand();

            //hostData = (float*)calloc(n, sizeof(float));

            checkCudaErrors(cudaMalloc((void **)&deviceData, sizeof(float) * n));
            CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, _seed));
            CURAND_CALL(curandGenerateUniform(gen, deviceData, n));
            s->stopTimer();

            //checkCudaErrors(cudaMemcpy(hostData, deviceData, n*sizeof(float), cudaMemcpyDeviceToHost));
            CURAND_CALL(curandDestroyGenerator(gen));
            checkCudaErrors(cudaFree(deviceData));
            //free(hostData);
            total_device += s->getTimeInMilliseconds();
        }

        std::cerr<<"For n = "<<n<<", curand using Host API took, on average: "<<total_device/static_cast<double>(100)<<std::endl;
        total_device = 0;

        n*=1000;
    }

    std::cerr<<std::endl<<"Beginning cuRand (device API), device random generation..."<<std::endl;
    n = 10;
    total_device = 0;

    int threadsPerBlock = 1024;

    for (size_t t =0; t < 2; ++t)
    {
        for (size_t i =0; i < 1000; ++i)
        {
            curandState* deviceStates;
            float* d_random_floats;
            int nBlocks = n/threadsPerBlock + 1;

            cudaSyncAndCheck();

            s->startTimer();

            checkCudaErrors(cudaMalloc(&deviceStates, n*sizeof(curandState)));
            initialise_curand_on_kernels<<<nBlocks, threadsPerBlock>>>(deviceStates, unsigned(time(NULL)), n);
            cudaSyncAndCheck();

            checkCudaErrors(cudaMalloc((void**) &d_random_floats, sizeof(float)* n));
            set_random_number_from_kernels<<<nBlocks, threadsPerBlock>>>(d_random_floats, deviceStates, n);
            cudaSyncAndCheck();

            s->stopTimer();
            total_device += s->getTimeInMilliseconds();


            //float* h_random_floats = (float *)malloc(sizeof(float) * n);
            //memset(h_random_floats, 0, sizeof(float) * n);
            //checkCudaErrors(cudaMemcpy(h_random_floats, d_random_floats, sizeof(float)*n, cudaMemcpyDeviceToHost));
            //free(h_random_floats);


            checkCudaErrors(cudaFree(d_random_floats));
            checkCudaErrors(cudaFree(deviceStates));

            deviceStates = NULL;
            d_random_floats = NULL;

        }

        std::cerr<<"For n = "<<n<<", curand using device API took, on average: "<<total_device/static_cast<double>(100)<<std::endl;
        total_device = 0;

        n*=1000;
    }

    utils::removeTimer(&s);

    cudaSyncAndCheck();
    checkCudaErrors(cudaDeviceReset());
    cudaSyncAndCheck();

}

int main(void)
{
  testRandomGeneration();
}
