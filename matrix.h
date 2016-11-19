#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <iostream> 
#include <iomanip>
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "util.h" 

using namespace std;

class Matrix {
  private: 
    int width; 
    int height; 
    float *array; 
    float *devArray;
  public: 
    Matrix(int width, int height, float *array) { 
      this->width = width; 
      this->height = height; 

      // allocate space for matrix on CPU 
      this->array = (float *)malloc(width * height * sizeof(*array)); 
      check(!this->array, "host memory allocation failed"); 

      // allocate space for matrix on GPU 
      cudaError_t cudaStat = cudaMalloc((void**)&devArray, 
          width*height*sizeof(*devArray)); 
      check(cudaStat != cudaSuccess, "device memory allocation failed"); 

      // copy matrix to GPU 
      cublasStatus_t stat = cublasSetMatrix(width, height, sizeof(*array), 
          array, width, devArray, width); 
      check(stat != CUBLAS_STATUS_SUCCESS, "data download failed"); } 

    void print() const { 
      // copy matrix to CPU
      cublasStatus_t stat = cublasGetMatrix(width, height, sizeof(*array), 
          devArray, width, array, width);
      check(stat != CUBLAS_STATUS_SUCCESS, "data download failed");

      int i, j;
      range(j, 0, height) {
        range(i, 0, width) {
          cout << setw(7) << array[idx2c(i,j,width)];
        }
        cout << endl;
      }
    }
};

ostream& operator<<(ostream& os, const Matrix& matrix)  
{
    matrix.print();
    return os;  
}

