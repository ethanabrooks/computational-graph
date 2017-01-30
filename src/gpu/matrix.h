#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix_struct {
   unsigned height;
   unsigned width;
   float *array;
} Matrix;

extern "C" {
  extern cublasHandle_t handle;

  void alloc_matrix(Matrix *matrix);
  int size(const Matrix *m);
  void get_array(const Matrix *matrix, float *dst);
  void set_array(Matrix *matrix, const float *array, bool transpose);
  void copy_matrix(const Matrix *src, Matrix *dst);
  void init_matrix(Matrix *matrix, const float *array,
          unsigned height, unsigned width);
  void fill_matrix(Matrix *matrix, float value);
  void print_matrix(const Matrix *matrix);
  void free_matrix(Matrix *matrix);

  //// allocates on device
  //void alloc_matrix(Matrix *matrix, int height, int width);

  //int size(const Matrix *m);
  //void init_cublas();
  //void copy_matrix(const Matrix *src, Matrix *dst);
  //void init_matrix(Matrix *matrix, const float *array, int height, int width);
  //void fill_matrix(Matrix *matrix, float value);
  //void print_matrix(const Matrix *matrix);
  //void download_matrix(const Matrix *src, float *dst);
  //void upload_matrix(const float *src, Matrix *dst);
  //void free_matrix(Matrix *matrix);
  //void maybe_init_cublas();
}

#endif
