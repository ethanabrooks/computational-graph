#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "matrix.h" 
#include "scan.h" 

bool reduce_equal_cpu(const Matrix *m, float x) {
  float *temp = (float*)malloc(size(m) * sizeof(*temp));
  check(!temp, "malloc failed for `temp` in `reduce_eq`");

  download_matrix(m, temp);
  bool all_eq = true;
  int i;
  rng(i, 0, size(m)) {
    if (temp[i] != x) {
      all_eq = false;
      break;
    }
  }

  free(temp);
  return all_eq;
}

float reduce_sum_scan(const Matrix *m) {
  int size_matrix = size(m);
  check(size_matrix == 0, "matrix must have more than 0 elements.");

  // temp buffer stores result of scan
  float *dev_temp;
  cudaError_t cudaStat = cudaMalloc(&dev_temp,
      size_matrix*sizeof(*dev_temp));
  check(cudaStat != cudaSuccess, "cudaMalloc failed for `temp` in `reduce_avg`");

  dev_scan(size_matrix, m->dev_array, dev_temp);

  // last element of scan is sum of all but last element of matrix
  float last_scan_val, last_matrix_val;

  cudaMemcpy(&last_scan_val, &dev_temp[size_matrix - 1], 
      sizeof(last_scan_val), cudaMemcpyDeviceToHost);

  cudaMemcpy(&last_matrix_val, &m->dev_array[size_matrix - 1],
      sizeof(last_matrix_val), cudaMemcpyDeviceToHost);

  cudaFree(dev_temp);
  return last_matrix_val + last_scan_val;
}
