#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"

__global__ void kernUpSweep(int n, int d, const float *idata, float *odata) {
  if (IDx >= n) return;
  int addTerm = (IDx + 1) % (d * 2) == 0 ? idata[IDx - d] : 0;
  odata[IDx] = idata[IDx] + addTerm;
}

__global__ void kernDownSweep(int length, int d, const float *idata, float *odata) {
  if (IDx >= length) return;

  // On the first iteration, and using only one thread, set the last element to 0.
  if ((IDx + 1) % d == 0) {
    int swapIndex = IDx - (d / 2);
    int term = (length == d) && (IDx == d - 1) ? 0 : idata[IDx];
    odata[IDx] = term + idata[swapIndex];
    odata[swapIndex] = term;
  }
}

void dev_scan(int n, const float *dev_idata, float *dev_odata) {

  // round n up to the nearest power of 2
  int bufferedLength = pow(2, ceil(log2((float)n))); 
  dim3 numBlocks = blockcount(bufferedLength); // enough blocks to allocate one thread to each array element

  float *dev_temp;
  cudaError_t cudaStat = cudaMalloc((void**)&dev_temp,
      bufferedLength*sizeof(*dev_temp));
  check(cudaStat != cudaSuccess, "cudaMalloc failed for `temp` in `reduce_avg`");
  cudaMemcpy(dev_temp, dev_idata, n * sizeof(*dev_temp),
      cudaMemcpyDeviceToDevice);

  // upsweep
  for (int d = 1; d <= n; d *= 2) {
    kernUpSweep <<<numBlocks, BLOCKSIZE >>>(n, d, dev_temp, dev_odata);
    std::swap(dev_temp, dev_odata);
  }

  // downsweep
  for (int d = bufferedLength; d >= 1; d /= 2) {
    kernDownSweep << <numBlocks, BLOCKSIZE >> >(bufferedLength, d, dev_temp, dev_odata);
    std::swap(dev_temp, dev_odata);
  }
}
