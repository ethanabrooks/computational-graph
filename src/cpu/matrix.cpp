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
  void maybe_init_cublas() {}

  int size(const Matrix *m) { return m->width * m->height; }

  // allocates on device
  void alloc_matrix(Matrix *matrix) {
    printf("size(matrix): %d\n", size(matrix));
    matrix->array = safe_malloc<float>(size(matrix));
  }

  float* get_array(const Matrix *matrix) {
    return matrix->array;
  }

  void set_array(Matrix *matrix, const float *array, bool transpose) {
    if (transpose) {
        int i;
        rng(i, 0, size(matrix)) {
            int iT = (i % matrix->height) * matrix->width + int(i / matrix->height);
            matrix->array[i] = array[iT];
        }
    } else {
        memcpy(matrix->array, array, size(matrix) * sizeof(float));
    }
  }

  void fill_matrix(Matrix *matrix, float value) {
    int i;
    rng(i, 0, size(matrix)) {
      matrix->array[i] = value;
    }
  }

  void init_matrix(Matrix *matrix, const float *array, 
          unsigned height, unsigned width) {
    matrix->height = height;
    matrix->width = width;
    alloc_matrix(matrix);
    set_array(matrix, array, true);
  }

  void copy_matrix(const Matrix *src, Matrix *dst) {
    assert(dst->height == src->height);
    assert(dst->width == src->width);
    set_array(dst, src->array, false);
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
