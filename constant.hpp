#ifndef CONSTANT_HPP
#define CONSTANT_HPP
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <cuda_runtime.h> 
#include <boost/optional.hpp>
#include "cublas_v2.h" 
#include "util.hpp" 
#endif

class Constant {
  private:
    virtual std::ostream& write(std::ostream& os) = 0;
  public:
    virtual Constant* operator+(const Constant &c) = 0;

  friend std::ostream& operator<< (std::ostream& stream, Constant& constant) {
    constant.write(stream);
  }
};

class Float: public Constant {
  private:
    float value;
  public:
    Float(float x) { value = x; }

    Float& operator+(const Constant &x) {  //HELP!!!
      return Float(value + x.value); 
    }
    std::ostream& write(std::ostream& os) {
      os << value;
      return os;
    }
};
