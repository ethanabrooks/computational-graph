#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <time.h>
#include "cublas_v2.h" 
#include "matrix.h" 
#include "op.h" 
#include "util.h" 

#define M 3
#define N 2

// globals
float *input;
float *devPtrInput;
float *weights;
float *devPtrWeights;
cublasHandle_t handle;


int main (void){ 

  // set values of input matrix
  float input_vals [] = {
    1, 2, 
    3, 4
  };

  // set values of weights matrix
  float weights_vals [] = {
    2, 2, 
    2, 2,
    2, 2,
  };

  cublasHandle_t handle;
  int i, j;
  float *array;
  Matrix matrix;
  Matrix m1;
  Matrix m2;

  array = (float *)malloc (M * N * sizeof (float)); 
  check(!array, "host memory allocation failed"); 

  rng(j, 0, N) {
    rng(i, 0, M) {
      array[idx2c(i,j,M)] = (float)idx2c(j, i, M);
    }
  }

  cudaError_t cudaStat;
  cublasStatus_t stat;

  stat = cublasCreate(&handle);
  check(stat != CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed"); 

  init_matrix(&matrix, array, M, N);
  alloc_matrix(&m1, M, N);
  alloc_matrix(&m2, M, N);
  fill_matrix(&m1, 2);
  fill_matrix(&m2, 3);
  elemwise_add(&m1, &m2, &matrix);
  print_matrix(&matrix);
  printf("\n");
  elemwise_mult(&m1, &m2, &matrix);
  print_matrix(&matrix);
  printf("\n");
  broadcast_add(1, &m2, &matrix);
  print_matrix(&matrix);
  printf("\n");
  fill_matrix(&m1, 2);
  init_matrix(&m1, weights_vals, M, N);
  print_matrix(&m1);
  printf("\n");

  printf("reduce_equal = %d\n", reduce_equal(&m1, 2.0));
  printf("reduce_sum = %f\n", reduce_sum(&m1));
  clock_t start = clock(), diff;
  /////
  //reduce_sum(&m1);
  /////
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC; 
  //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  return EXIT_SUCCESS;
}
