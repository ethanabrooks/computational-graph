#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 

extern "C" {
  dim3 blockcount(int count) {
    float numblocks = (count / BLOCKSIZE.x + 1);
    return pow(2, ceil(log2(numblocks))); \
  }

  void check(int condition, const char *msg) {
    if (condition) {
      fprintf(stderr, "ERROR: %s\n", msg);
      exit(EXIT_FAILURE);
    }
  }

  int size(const Matrix *m) { return m->width * m->height; }

  __global__ 
  void _memset(int len, float *array, float value) {
    SET(array, value);
  }

  // allocates on device
  void alloc_matrix(Matrix *matrix, int height, int width) { 
    matrix->width = width;
    matrix->height = height;

    // allocate space for matrix on GPU 
    cudaError_t cudaStat = cudaMalloc((void**)&matrix->dev_array, 
        width*height*sizeof(*matrix->dev_array)); 
    check(cudaStat != cudaSuccess, "device memory allocation failed"); 
  }

  void init_matrix(Matrix *matrix, float *array, int height, int width) {
    alloc_matrix(matrix, height, width);

    // copy matrix to GPU 
    cublasStatus_t stat = cublasSetMatrix(width, height, sizeof(*array), 
        array, width, matrix->dev_array, width); 
    check(stat != CUBLAS_STATUS_SUCCESS, "data upload failed"); 

    cudaMemcpy(array, matrix->dev_array,
        height * width * sizeof(*array),
        cudaMemcpyDeviceToHost);
  }

  void copy_matrix(Matrix *src, Matrix *dst) {
     
    // copy matrix from src
    cudaMemcpy(dst->dev_array, src->dev_array, 
        src->height * src->width * sizeof(*src->dev_array),
        cudaMemcpyDeviceToDevice);
  }

  void fill_matrix(Matrix *matrix, float value) {
    DEFAULT_LAUNCH(_memset, matrix, value)
  }

  void download_matrix(const Matrix *src, float *dst) {
    cublasStatus_t stat = cublasGetMatrix(src->width, src->height, 
        sizeof(*src->dev_array), 
        src->dev_array, src->width, dst, src->width);
    check(stat != CUBLAS_STATUS_SUCCESS, "data download failed");
  }

  void print_matrix(Matrix *matrix) {

    // allocate space for matrix on CPU 
    float *array = (float *)malloc(matrix->width * matrix->height *
        sizeof(*matrix->dev_array)); 
    check(!array, "host memory allocation failed"); 

    // copy matrix to CPU
    download_matrix(matrix, array);

    int i, j;
    rng(j, 0, matrix->height) {
      rng(i, 0, matrix->width) {
        printf("%7.0f", array[idx2c(i, j, matrix->width)]);
      }
      printf("\n");
    }

    free(array);
  }

  void free_matrix(Matrix *matrix) {
    free(matrix->dev_array);
  }
}

