#include "matrix.h" 

void elemwise_multiply(Matrix *m1, Matrix *m2, Matrix *result);
void elemwise_add(Matrix *m1, Matrix *m2, Matrix *result);
void matrix_neg(Matrix *m, Matrix *result);
void scalar_multiply(float val, Matrix *m, Matrix *result);
void scalar_add(float val, Matrix *m, Matrix *result);
