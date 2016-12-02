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

/*void device2host(float *src, float *dst) {*/
    /*cudaError_t cudaStat = cudaMemcpy(dev_sum, &z, sizeof(z), cudaMemcpyHostToDevice);*/
    /*check(cudaStat != cudaSuccess, "cudaMemcpy failed");*/
/*}*/
