#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 
#include "op.h" 

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
    1, 1, 
    1, 1
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
  broadcast_sub_rev(&m2, 2, &matrix);
  print_matrix(&matrix);
  printf("\n");
  printf("sum over m1: %f", reduce_sum(&m2));
  printf("\n");
  std::cout << "m1 == 2: " << reduce_equal(&m1, 2) << std::endl;

  return EXIT_SUCCESS;
}
