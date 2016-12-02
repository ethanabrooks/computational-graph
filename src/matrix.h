#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix_struct {
   int height;
   int width;
   float *dev_array;
} Matrix;

extern "C" {
  // allocates on device
  void alloc_matrix(Matrix *matrix, int height, int width);

  int size(const Matrix *m);
  void copy_dev2dev(Matrix *src, Matrix *dst);
  void copy_matrix(Matrix *src, Matrix *dst);
  void init_matrix(Matrix *matrix, float *array, int height, int width);
  void fill_matrix(Matrix *matrix, float value);
  void print_matrix(Matrix *matrix);
  void download_matrix(const Matrix *src, float *dst);
  void free_matrix(Matrix *matrix);
}

#endif
