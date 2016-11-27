#ifndef MATRIX_H
#define MATRIX_H
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 

#define range(i, start, end) for(i = start; i < end; i++)
#define idx2c(i, j, width) ((j) * (width) + (i))
#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define BLOCKSIZE dim3(16)
#define BLOCKSIZE2D dim3(16, 16)
#define SET(a, expr) if (IDx < len) { a[IDx] = expr; }
#define ZIP_WITH(op) SET(result, a1[IDx] op a2[IDx])
#define DEFAULT_LAUNCH(kernel, out, ...) \
  kernel<<<blockcount(size(*out)), BLOCKSIZE>>> \
  (size(*out), out->devArray, ##__VA_ARGS__)


int double_input(int input);

typedef struct matrix_struct {
   int height;
   int width;
   float *devArray;
} Matrix;

void check(int condition, const char *msg);
int size(Matrix m);
dim3 blockcount(int count);
void alloc_matrix(Matrix *matrix, int width, int height);
void init_matrix(Matrix *matrix, float *array, int width, int height);
void copy_matrix(Matrix *src, Matrix *dst);
void fill_matrix(Matrix *matrix, float value);
void print_matrix(Matrix *matrix);

#endif
