#include "matrix.h" 

extern "C" {

  void map_neg(Matrix *m, Matrix *result);

  void elemwise_mult(Matrix *m1, Matrix *m2, Matrix *result);
  void elemwise_add(Matrix *m1, Matrix *m2, Matrix *result);
  void elemwise_sub(Matrix *m1, Matrix *m2, Matrix *result);

  void broadcast_mult(float val, Matrix *m, Matrix *result); 
  void broadcast_add(float val, Matrix *m, Matrix *result); 
  void broadcast_sub(float val, Matrix *m, Matrix *result); // e.g. 1 - [ [1 1] [1 1] ]

  void broadcast_sub_rev(Matrix *m, float val, Matrix *result); // e.g. [ [1 1] [1 1] ] - 1
}

