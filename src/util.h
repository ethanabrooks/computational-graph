#ifndef UTIL_H
#define UTIL_H

#define rng(i, start, end) for(i = start; i < end; i++)
#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define BLOCKSIZE dim3(16)
#define BLOCKSIZE2D dim3(16, 16)
#define DEFAULT_LAUNCH(kernel, out, ...) \
  kernel<<<blockcount(size(out)), BLOCKSIZE>>> \
      (size(out), out->dev_array, ##__VA_ARGS__); 

void check(int condition, const char *msg);
dim3 blockcount(int count);

template<typename T> inline T* safe_malloc(int count) {
    T *array = (T *)malloc(count * sizeof(*array));
    check(!array, "safe_malloc failed"); 
    return array;
}

template<typename T> inline T* safe_cuda_malloc(int count) {
    T *array;
    cudaError_t cudaStat = cudaMalloc((void**)&array, count * sizeof(*array));
    check(cudaStat != cudaSuccess, "float_malloc_cuda failed");
    return array;
}

template<typename T> inline 
cudaError_t device2host(int count, const T *src, T *dst) {
    return cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyDeviceToHost);
}

template<typename T> inline 
cudaError_t host2device(int count, const T *src, T *dst) {
   return  cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice);
}

template<typename T> inline 
cudaError_t device2device(int count, const T *src, T *dst) {
    return cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyDeviceToDevice);
}

#endif
