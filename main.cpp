#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <iostream>
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "util.h" 
#include "matrix.h" 

#define M 2
#define N 2

using namespace std;

// globals
float *input;
float *devPtrInput;
float *weights;
float *devPtrWeights;
cublasHandle_t handle;

int main (void){ 

  cublasStatus_t custat;

  // set values of input matrix
  float input_vals [] = {
    1, 2, 
    3, 4
  };
  Matrix m(M, N, input_vals);

  // set values of weights matrix
  float weights_vals [] = {
    1, 1, 
    1, 1
  };

  Matrix n(M, N, input_vals);
  cout << n << endl;

  return EXIT_SUCCESS;
}
