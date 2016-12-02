#ifndef UTIL_H
#define UTIL_H

#define rng(i, start, end) for(i = start; i < end; i++)
#define idx2c(i, j, width) ((j) * (width) + (i))
#define IDx ((blockIdx.x * blockDim.x) + threadIdx.x)
#define IDy ((blockIdx.y * blockDim.y) + threadIdx.y)
#define BLOCKSIZE dim3(16)
#define BLOCKSIZE2D dim3(16, 16)
#define SET(result, value) if (IDx < len) { result[IDx] = value; }
#define DEFAULT_LAUNCH(kernel, out, ...) \
  kernel<<<blockcount(size(out)), BLOCKSIZE>>> \
      (size(out), out->dev_array, ##__VA_ARGS__); 

void check(int condition, const char *msg);
dim3 blockcount(int count);
float* float_malloc(int count); 

#endif
