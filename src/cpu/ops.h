#ifndef OP_H
#define OP_H

#include "matrix.h" 

extern "C" {
  void gemm(const Matrix *m1, bool trans1, const Matrix *m2, bool trans2, 
      Matrix *result);

  void map_neg(Matrix *m, Matrix *result);
  void map_sq(Matrix *m, Matrix *result);
  void map_abs(Matrix *m, Matrix *result);
  void map_signum(Matrix *m, Matrix *result);
  void map_sigmoid(Matrix *m, Matrix *result);
  void map_tanh(Matrix *m, Matrix *result);
  void map_one_minus(Matrix *m, Matrix *result);

  void elemwise_mul(Matrix *m1, Matrix *m2, Matrix *result);
  void elemwise_add(Matrix *m1, Matrix *m2, Matrix *result);
  void elemwise_sub(Matrix *m1, Matrix *m2, Matrix *result);

  void broadcast_mul(float val, Matrix *m, Matrix *result); 
  void broadcast_add(float val, Matrix *m, Matrix *result); 
  void broadcast_sub(float val, Matrix *m, Matrix *result); // e.g. 1 - [ [1 1] [1 1] ]

  bool all_equal(const Matrix *m, float x);
  bool all_less_than(const Matrix *m, float x);
  float reduce_sum(const Matrix *m);
}

#endif
