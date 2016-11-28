#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 


__global__ 
void _array_multiply(int len, float *result, float *a1, float *a2) {
  ZIP_WITH(*);
}

__global__ 
void _array_add(int len, float *result, float *a1, float *a2) {
  ZIP_WITH(+);
}

__global__ 
void _array_mult_scalar(int len, float *result, float *a, float val) {
  SET(result, val * a[IDx]);
}

__global__ 
void _array_add_scalar(int len, float *result, float *a, float val) {
  SET(result, val + a[IDx]);
}

__global__ 
void _array_neg(int len, float *result, float *a) {
  SET(result, -a[IDx]);
}

void check_dims(Matrix *m1, Matrix *m2, Matrix *result) { 
  check(m1->height != m2->height 
     || m1->width  != m2->width
     || m1->height != result->height 
     || m1->width  != result->width, 
      "matrices must have the same dimensions");
}

void elemwise_multiply(Matrix *m1, Matrix *m2, Matrix *result) { 
  check_dims(m1, m2, result);
  DEFAULT_LAUNCH(_array_multiply, result, m1->devArray, m2->devArray);
}

void elemwise_add(Matrix *m1, Matrix *m2, Matrix *result) { 
  check_dims(m1, m2, result);
  DEFAULT_LAUNCH(_array_add, result, m1->devArray, m2->devArray);
}

void scalar_multiply(float val, Matrix *m, Matrix *result) { 
  DEFAULT_LAUNCH(_array_mult_scalar, result, m->devArray, val);
}

void scalar_add(float val, Matrix *m, Matrix *result) { 
  DEFAULT_LAUNCH(_array_add_scalar, result, m->devArray, val);
}

void matrix_neg(Matrix *m, Matrix *result) { 
  DEFAULT_LAUNCH(_array_neg, result, m->devArray);
}

