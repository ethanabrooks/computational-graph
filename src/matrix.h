#ifndef MATRIX_H
#define MATRIX_H
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 

#define rng(i, start, end) for(i = start; i < end; i++)
#define idx2c(i, j, width) ((j) * (width) + (i))
#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define BLOCKSIZE dim3(16)
#define BLOCKSIZE2D dim3(16, 16)
#define SET(result, value) if (IDx < len) { result[IDx] = value; }
#define DEFAULT_LAUNCH(kernel, out, ...) \
  kernel<<<blockcount(size(*out)), BLOCKSIZE>>> \
  (size(*out), out->dev_array, ##__VA_ARGS__);

typedef struct matrix_struct {
   int height;
   int width;
   float *dev_array;
} Matrix;

extern "C" {
  // allocates on device
  void alloc_matrix(Matrix *matrix, int height, int width);

  void copy_matrix(Matrix *src, Matrix *dst);
  void init_matrix(Matrix *matrix, float *array, int height, int width);
  void check(int condition, const char *msg);
  int size(Matrix m);
  dim3 blockcount(int count);
  void fill_matrix(Matrix *matrix, float value);
  void print_matrix(Matrix *matrix);
  void download_array(Matrix *src, float *dst);
  void free_matrix(Matrix *matrix);
}

#endif
