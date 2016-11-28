#include "matrix.h" 

void map_neg(Matrix *m, Matrix *result);
void broadcast_mult(float val, Matrix *m, Matrix *result); 
void broadcast_add(float val, Matrix *m, Matrix *result); 
void elemwise_mult(Matrix *m1, Matrix *m2, Matrix *result);
void elemwise_add(Matrix *m1, Matrix *m2, Matrix *result);
