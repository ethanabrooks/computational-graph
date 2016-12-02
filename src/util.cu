#include <stdio.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "util.h" 

dim3 blockcount(int count) {
  float numblocks = (count / BLOCKSIZE.x + 1);
  return pow(2, ceil(log2(numblocks)));
}

void check(int condition, const char *msg) {
  if (condition) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
  }
}

float* float_malloc(int count) {
    float *array = (float *)malloc(count * sizeof(float));
    check(!array, "host memory allocation failed"); 
    return array;
}
