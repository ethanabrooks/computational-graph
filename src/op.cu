#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 
#include "util.h" 

#define UN_MAP(name, f_body) \
  __device__ \
  float f_ ## name(float x) { \
    return f_body; \
  } \
  __global__ \
  void _ ## name(int len, float *result, const float *a) { \
    SET(result, f_ ## name(a[IDx])) \
  } \
  void map_ ## name(const Matrix *m, Matrix *result) { \
    DEFAULT_LAUNCH(_ ## name, result, m->dev_array); \
  }

#define BIN_BROADCAST(name, op) \
  __global__ \
  void _ ## name ## _scalar(int len, float *result, const float *a, float val) { \
    SET(result, val op a[IDx]) \
  } \
  void broadcast_ ## name(float val, const Matrix *m, Matrix *result) { \
    DEFAULT_LAUNCH(_ ## name ## _scalar, result, m->dev_array, val); \
  }

#define BIN_BROADCAST_REV(name, op) \
  __global__ \
  void _ ## name ## _scalar_rev(int len, float *result, const float *a, float val) { \
    printf("TEST TEST TEST\n"); \
    SET(result, a[IDx] op val) \
  } \
  void broadcast_ ## name ## _rev(const Matrix *m, float val, Matrix *result) { \
    DEFAULT_LAUNCH(_ ## name ## _scalar_rev, result, m->dev_array, val); \
  }

#define BIN_ELEMWISE(name, op) \
  __global__ \
  void _ ## name (int len, float *result, const float *a1, const float *a2) { \
    SET(result, a1[IDx] op a2[IDx]) \
  } \
  void elemwise_ ## name (const Matrix *m1, const Matrix *m2, Matrix *result) { \
    check_dims(m1, m2, result); \
    DEFAULT_LAUNCH(_ ## name, result, m1->dev_array, m2->dev_array); \
  }

void check_dims(const Matrix *m1, const Matrix *m2, const Matrix *result) { 
  check(m1->height != m2->height 
     || m1->width  != m2->width
     || m1->height != result->height 
     || m1->width  != result->width, 
      "matrices must have the same dimensions");
}


extern "C" {
  UN_MAP(neg, -x) // map_neg
  UN_MAP(abs, x < 0 ? -x : x) // map_aps
  UN_MAP(signum, x < 0 ? -1 : 1) // map_signum
  UN_MAP(sigmoid, 1.0f / (1.0f + expf(-x))) // map_sigmoid

  BIN_ELEMWISE(mult, *) // elemwise_mult
  BIN_ELEMWISE(add, +) // elemwise_add
  BIN_ELEMWISE(sub, -) // elemwise_sub

  BIN_BROADCAST(mult, *) // broadcast_mult
  BIN_BROADCAST(add, +) // broadcast_add
  BIN_BROADCAST(sub, -) // broadcast_sub

  BIN_BROADCAST_REV(sub, -) // broadcast_sub_rev

  __global__
  void _reduce_equal(int len, const float *a, unsigned int *boolean, float x) {
    if (IDx >= len) return;
    atomicAnd(boolean, a[IDx] == x); 
  }

  __global__
  void _reduce_sum(int len, const float *a, float *sum) {
    if (IDx >= len) return;
    atomicAdd(sum, a[IDx]); 
  }

  bool reduce_equal(const Matrix *m, float x) {
    unsigned int *dev_bool = safe_cuda_malloc<unsigned int>(1);
    unsigned int t = 1;

    cudaError_t cudaStat = host2device(1, &t, dev_bool);
    check(cudaStat != cudaSuccess, "host2device failed in reduce_eq");

    _reduce_equal<<<blockcount(size(m)), BLOCKSIZE>>> 
      (size(m), m->dev_array, dev_bool, x);

    cudaStat = device2host(1, dev_bool, &t);
    check(cudaStat != cudaSuccess, "device2host failed in reduce_sum");

    cudaFree(dev_bool);
    return t == 1;
  }

  float reduce_sum(const Matrix *m) {
    float *dev_sum = safe_cuda_malloc<float>(1);
    float sum = 0;

    cudaError_t cudaStat = host2device(1, &sum, dev_sum);
    check(cudaStat != cudaSuccess, "host2device failed in reduce_sum");

    _reduce_sum<<<blockcount(size(m)), BLOCKSIZE>>>
      (size(m), m->dev_array, dev_sum);

    cudaStat = device2host(1, dev_sum, &sum);
    check(cudaStat != cudaSuccess, "device2host failed in reduce_sum");

    cudaFree(dev_sum);
    return sum;
  }
}
