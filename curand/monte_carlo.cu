#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <iostream>
#include <iomanip>

// we could vary M & N to find the perf sweet spot

struct estimate_pi_curand : public thrust::unary_function<unsigned int, float>
{
  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 100000; // samples per thread

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float x = curand_uniform(&s);
      float y = curand_uniform(&s);

      // measure distance from the origin
      float dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};


__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}


struct estimate_pi_thrust : public thrust::unary_function<unsigned int,float>
{
  __host__ __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 100000; // samples per thread

    //unsigned int seed = hash(thread_id);
    unsigned int seed = thread_id;

    // seed a random number generator
    thrust::default_random_engine rng(seed);

    // create a mapping from random numbers to [0,1)
    thrust::uniform_real_distribution<float> u01(0,1);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float x = u01(rng);
      float y = u01(rng);

      // measure distance from the origin
      float dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

void getThrustTime(const unsigned M)
{
  boost::posix_time::ptime start(
      boost::posix_time::microsec_clock::universal_time()); 
  float estimate = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M),
        estimate_pi_thrust(),
        0.0f,
        thrust::plus<float>());
  boost::posix_time::ptime end(
      boost::posix_time::microsec_clock::universal_time());
  estimate /= M;
  std::cout << std::setprecision(7);
  std::cout << "pi is approximately ";
  std::cout << estimate << " thrust time:" << (end-start) << std::endl;
}

void getCurandTime(const unsigned M)
{
  boost::posix_time::ptime start(
      boost::posix_time::microsec_clock::universal_time()); 
  float estimate = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M),
        estimate_pi_curand(),
        0.0f,
        thrust::plus<float>());
  boost::posix_time::ptime end(
      boost::posix_time::microsec_clock::universal_time());
  estimate /= M;
  std::cout << std::setprecision(7);
  std::cout << "pi is approximately ";
  std::cout << estimate << " curand time:" << (end-start) << std::endl;
}

int main(void)
{
  unsigned M(3000000);
  getCurandTime(M);
  getThrustTime(M);
  return 0;
}

