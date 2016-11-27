#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
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

  range(j, 0, N) {
    range(i, 0, M) {
      array[idx2c(i,j,M)] = (float)idx2c(j, i, M);
    }
  }

  cudaError_t cudaStat;
  cublasStatus_t stat;

  stat = cublasCreate(&handle);
  check(stat != CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed"); 

  new_matrix(&matrix, array, M, N);
  new_matrix(&m1, array, M, N);
  new_matrix(&m2, array, M, N);
  fill_matrix(&m1, 2);
  fill_matrix(&m2, 3);
  elemwise_multiply(&m1, &m2, &matrix);
  print_matrix(&matrix);

  elemwise_add(&m1, &m2, &matrix);

  print_matrix(&matrix);
  return EXIT_SUCCESS;
}
