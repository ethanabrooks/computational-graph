#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <assert.h>
#include <math.h> 
#include <algorithm> 
#include "matrix.h" 
#include "util.h" 

#define MAT_IDX(i, j, width) ((i) * (width) + (j))

extern "C" {

  int size(const Matrix *m) { return m->width * m->height; }

  void download_matrix(const Matrix *src, float *dst) {
      dst = src->array;
  }

  void upload_matrix(const float *src, Matrix *dst) {
      memcpy(dst->array, src, size(dst) * sizeof(*src));
  }

  // allocates on device
  void alloc_matrix(Matrix *matrix, int height, int width) { 
    matrix->width = width;
    matrix->height = height;
    matrix->array = safe_malloc<float>(size(matrix));
  }

  void init_matrix(Matrix *matrix, const float *array, int height, int width) {
    alloc_matrix(matrix, height, width);
    int i;
    rng(i, 0, size(matrix)) {
        int iT = (i % matrix->height) * matrix->width + int(i / matrix->height);
        matrix->array[i] = array[iT];
    }
  }

  void copy_matrix(const Matrix *src, Matrix *dst) {
    assert(size(dst) == size(src));
    dst->height = src->height;
    dst->width = src->width;
    memcpy(dst->array, src->array, size(src) * sizeof(float));
  }

  void fill_matrix(Matrix *matrix, float value) {
    int i;
    rng(i, 0, size(matrix)) {
      matrix->array[i] = value;
    }
  }


  void print_matrix(const Matrix *matrix) {
    int i, j;
    rng(i, 0, matrix->height) {
      rng(j, 0, matrix->width) {
        printf("%7.2f", matrix->array[MAT_IDX(j, i, matrix->height)]);
      }
      printf("\n");
    }
  }

  void free_matrix(Matrix *matrix) {
      free(matrix->array);
  }
}
