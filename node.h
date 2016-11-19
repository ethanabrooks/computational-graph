#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include "util.h" 
#include "matrix.h" 

class Constant {
  private:
    enum class { float_t, matrix_t } type;
    float floatValue;
    Matrix matrixValue;
  public:
    Constant(float x) {
      type = type::float_t;
      floatValue = x;
    }

    Constant(Matrix m) {
      type = type::matrix_t;
      matrixValue = m;
    }

    void print() const {
      switch (type) {
        case floatType:
          cout << floatValue;
        case matrixType:
          cout << matrixValue;
      }
    }
};

class Node {
  private: 
    Constant value;
};
