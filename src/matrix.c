#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "util.h" 

int double_input(int input) {
    return input * 2;
}

typedef struct matrix_struct {
   int width;
   int height;
   float *devArray;
   float *array;
} Matrix;

void new_matrix(Matrix *matrix, float *array, int width, int height) {
  cudaError_t cudaStat;
  cublasStatus_t stat;

  matrix->array = (float *)malloc(width * height * sizeof(*matrix->array)); 
  check(!matrix->array, "host memory allocation failed"); 

  // allocate space for matrix on GPU 
  cudaStat = cudaMalloc((void**)&matrix->devArray, 
      width*height*sizeof(*matrix->array)); 
  check(cudaStat != cudaSuccess, "device memory allocation failed"); 

  // copy matrix to GPU 
  stat = cublasSetMatrix(width, height, sizeof(*array), 
      array, width, matrix->devArray, width); 
  check(stat != CUBLAS_STATUS_SUCCESS, "data upload failed"); 
}

void print_matrix(Matrix *matrix, int width, int height) {
  cublasStatus_t stat;

  // copy matrix to CPU
  stat = cublasGetMatrix(width, height, sizeof(*matrix->array), 
      matrix->devArray, width, matrix->array, width);
  check(stat != CUBLAS_STATUS_SUCCESS, "data download failed");

  int i, j;
  range(j, 0, height) {
    range(i, 0, width) {
      printf("%7.0f", matrix->array[idx2c(i,j,width)]);
    }
    printf("\n");
  }
}

