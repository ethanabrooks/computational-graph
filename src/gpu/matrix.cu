#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 
#include "util.h" 

#define MAT_IDX(i, j, width) ((i) * (width) + (j))

cublasHandle_t handle = NULL;

extern "C" {
  void maybe_init_cublas() {
    cublasStatus_t stat = cublasCreate(&handle);
    check(stat != CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed"); 
  };

  int size(const Matrix *m) { return m->width * m->height; }

  void download_matrix(const Matrix *src, float *dst) {
    cublasStatus_t stat = cublasGetMatrix(src->width, src->height, 
        sizeof(*src->dev_array), 
        src->dev_array, src->width, dst, src->width);
    check(stat != CUBLAS_STATUS_SUCCESS, "download_matrix failed");
  }


  __global__ 
  void _transpose(int len, float *out, const float *in, int lda, int ldb) {
    SET(out, in[(IDx % lda) * ldb + int(IDx / lda)])
  }

  void upload_matrix(const float *src, Matrix *dst) {
    float *temp = safe_cuda_malloc<float>(size(dst));
    host2device<float>(size(dst), src, temp);
    DEFAULT_LAUNCH(_transpose, dst, temp, dst->height, dst->width);
    cudaFree(temp);
  }

  // allocates on device
  void alloc_matrix(Matrix *matrix, int height, int width) { 
    if (!handle) {
      cublasStatus_t stat = cublasCreate(&handle);
      check(stat != CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed"); 
    }
    matrix->width = width;
    matrix->height = height;

    // allocate space for matrix on GPU 
    cudaError_t cudaStat = cudaMalloc((void**)&matrix->dev_array, 
        width*height*sizeof(*matrix->dev_array)); 
    check(cudaStat != cudaSuccess, "device memory allocation failed"); 
  }

  void init_matrix(Matrix *matrix, const float *array, int height, int width) {
    alloc_matrix(matrix, height, width);
    upload_matrix(array, matrix);
  }

  void copy_matrix(const Matrix *src, Matrix *dst) {
    dst->height = src->height;
    dst->width = src->width;

    cudaError_t stat = device2device<float>(size(src), 
        src->dev_array, dst->dev_array);
    check(stat != cudaSuccess, "copy_matrix failed");
  }

  __global__ 
  void _memset(int len, float *array, float value) {
    SET(array, value);
  }

  void fill_matrix(Matrix *matrix, float value) {
    DEFAULT_LAUNCH(_memset, matrix, value);
  }


  void print_matrix(const Matrix *matrix) {
  // allocate space for matrix on CPU 
  float *array = safe_malloc<float>(size(matrix));

  // copy matrix to CPU
  download_matrix(matrix, array);

  int i, j;
  rng(i, 0, matrix->height) {
    rng(j, 0, matrix->width) {
      printf("%7.0f", array[MAT_IDX(j, i, matrix->height)]);
    }
    printf("\n");
  }

    free(array);
  }

  void free_matrix(Matrix *matrix) {
    cudaFree(matrix->dev_array);
  }
}

