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
#include "ops.h" 
#include "util.h" 

#define L 2
#define M 3
#define N 2

// globals
float *input;
float *devPtrInput;
float *weights;
float *devPtrWeights;

int main (void){ 

  // set values of input matrix
  float input_vals [] = {
    1, 2, 
    3, 4
  };

  // set values of weights matrix
  float weights_vals [] = {
    1, 2, 
    3, 4,
    5, 6
  };

  int i, j;
  float *array;
  Matrix m1;
  Matrix m2;
  Matrix result;

  array = (float *)malloc (M * N * sizeof (float)); 
  check(!array, "host memory allocation failed"); 

  init_cublas();
  //alloc_matrix(&m1, 2, 2);
  //alloc_matrix(&m2, 3, 2);
  //alloc_matrix(&result, 3, 3);

  //init_matrix(&m1, input_vals, 2, 2);
  init_matrix(&m2, weights_vals, 3, 2);
  //print_matrix(&m2);
  //printf("reduce_sum = %f\n", reduce_sum(&m2));
  //fill_matrix(&m2, 1);
  //fill_matrix(&m2, 2);

  //fill_matrix(&result, 0);
  //print_matrix(&m2);
  //printf("\n");
  //print_matrix(&m2);
  //print_matrix(&result);
  //printf("\n");
  //gemm(&m1, false, &m2, false, &result);
  //print_matrix(&result);
  //fill_matrix(&m2, 3);
  //elemwise_add(&m1, &m2, &matrix);
  //printf("\n");
  //elemwise_mult(&m1, &m2, &matrix);
  //print_matrix(&matrix);
  //printf("\n");
  //broadcast_add(1, &m2, &matrix);
  //print_matrix(&matrix);
  //printf("\n");
  //fill_matrix(&m1, 2);
  //init_matrix(&m1, weights_vals, M, N);
  //print_matrix(m1);
  //copy_matrix(m1, m2);
  print_matrix(&m2);
  //printf("\n");

  printf("reduce_lt = %d\n", reduce_lt(&m2, 6.1));
  //clock_t start = clock(), diff;
  ///////
  ////reduce_sum(&m1);
  ///////
  //diff = clock() - start;
  //int msec = diff * 1000 / CLOCKS_PER_SEC; 
  //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
