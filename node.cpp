#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <cuda_runtime.h> 
#include <boost/optional.hpp>
#include "cublas_v2.h" 
#include "util.h" 
#include "matrix.h" 

using namespace std;
using namespace boost;

class Constant {
  private:
    enum class Type { float_t, matrix_t } type;
    optional<float> floatValue;
    optional<Matrix> matrixValue;
  public:
    Constant(float x) {
      type = Type::float_t;;
      floatValue = x;
    }

    Constant(Matrix m) {
      type = Type::matrix_t;;
      matrixValue = m;
    }

    void print() const {
      switch (this->type) {
        case Type::float_t:
          cout << floatValue;
        case Type::matrix_t:
          cout << matrixValue;
      }
    }
};

class Node {
  private: 
    Constant value;
};
