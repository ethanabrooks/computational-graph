#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 

int double_input(int input) {
  return 2 * input;
}

dim3 blockcount(int count) {
  return dim3(count / BLOCKSIZE.x + 1);
}

void check(int condition, const char *msg) {
  if (condition) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
  }
}

int size(Matrix m) { return m.width * m.height; }

__global__ 
void _memset(int len, float *array, float value) {
  SET(array, value);
}

void alloc_matrix(Matrix *matrix, int height, int width) {
  matrix->width = width;
  matrix->height = height;

  // allocate space for matrix on GPU 
  cudaError_t cudaStat = cudaMalloc((void**)&matrix->devArray, 
      width*height*sizeof(*matrix->devArray)); 
  check(cudaStat != cudaSuccess, "device memory allocation failed"); 
}

void init_matrix(Matrix *matrix, float *array, int height, int width) {
  alloc_matrix(matrix, height, width);

  // copy matrix to GPU 
  cublasStatus_t stat = cublasSetMatrix(width, height, sizeof(*array), 
      array, width, matrix->devArray, width); 
  check(stat != CUBLAS_STATUS_SUCCESS, "data upload failed"); 

  cudaMemcpy(array, matrix->devArray, height * width * sizeof(*array), cudaMemcpyDeviceToHost);
}

void copy_matrix(Matrix *src, Matrix *dst) {
  alloc_matrix(dst, src->height, src->width);

  // copy matrix from src
  cudaMemcpy(dst->devArray, src->devArray, 
      src->height * src->width * sizeof(*src->devArray), cudaMemcpyDeviceToDevice);
}

void fill_matrix(Matrix *matrix, float value) {
  DEFAULT_LAUNCH(_memset, matrix, value);
}

void print_matrix(Matrix *matrix) {
  cublasStatus_t stat;

  // allocate space for matrix on CPU 
  float *array = (float *)malloc(matrix->width * matrix->height *
      sizeof(*matrix->devArray)); 
  check(!array, "host memory allocation failed"); 


  // copy matrix to CPU
  stat = cublasGetMatrix(matrix->width, matrix->height, sizeof(*matrix->devArray), 
      matrix->devArray, matrix->width, array, matrix->width);
  check(stat != CUBLAS_STATUS_SUCCESS, "data download failed");

  int i, j;
  range(j, 0, matrix->height) {
    range(i, 0, matrix->width) {
      printf("%7.0f", array[idx2c(i, j, matrix->width)]);
    }
    printf("\n");
  }
}

