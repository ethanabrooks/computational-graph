#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix_struct {
   unsigned height;
   unsigned width;
   float *array;
} Matrix;

extern "C" {
  void alloc_matrix(Matrix *matrix, int height, int width);

  int size(const Matrix *m);
  void copy_matrix(const Matrix *src, Matrix *dst);
  void init_matrix(Matrix *matrix, const float *array, int height, int width);
  void fill_matrix(Matrix *matrix, float value);
  void print_matrix(const Matrix *matrix);
  void free_matrix(Matrix *matrix);
}

#endif
