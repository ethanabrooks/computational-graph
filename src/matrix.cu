#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 
#include "util.h" 

extern "C" {
  __global__ 
  void _memset(int len, float *array, float value) {
    SET(array, value);
  }

  int size(const Matrix *m) { return m->width * m->height; }

  void copy_dev2dev(Matrix *src, Matrix *dst) {
      cudaError stat = cudaMemcpy(dst->dev_array, src->dev_array, 
          src->height * src->width * sizeof(*src->dev_array),
          cudaMemcpyDeviceToDevice);
      check(stat != cudaSuccess, "copy_dev2dev failed");
  }

  void download_matrix(const Matrix *src, float *dst) {
    cublasStatus_t stat = cublasGetMatrix(src->width, src->height, 
        sizeof(*src->dev_array), 
        src->dev_array, src->width, dst, src->width);
    check(stat != CUBLAS_STATUS_SUCCESS, "download_matrix failed");
  }

  void upload_matrix(float *src, const Matrix *dst) {
    cublasStatus_t blas_stat = cublasSetMatrix(dst->width, dst->height, 
        sizeof(*src), src, dst->width, dst->dev_array, dst->width); 
    check(blas_stat != CUBLAS_STATUS_SUCCESS, "upload_matrix failed"); 

    /*cudaError custat = cudaMemcpy(src, dst->dev_array,*/
        /*size(dst) * sizeof(*src), cudaMemcpyDeviceToHost);*/
    /*check(custat != cudaSuccess, "data upload failed"); */
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
    upload_matrix(array, matrix);
  }

  void copy_matrix(Matrix *src, Matrix *dst) {
  }

  void fill_matrix(Matrix *matrix, float value) {
    DEFAULT_LAUNCH(_memset, matrix, value)
  }

  void print_matrix(Matrix *matrix) {

    // allocate space for matrix on CPU 
    float *array = safe_malloc<float>(size(matrix));

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
    cudaFree(matrix->dev_array);
  }
}

