
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <cuda_runtime.h> 
#include <boost/optional.hpp>
#include "cublas_v2.h" 
#include "util.hpp" 

class A {
  public:
    std::ostream& operator<< (std::ostream& os) {
      return os;  
    }
};

class B {
    std::ostream& operator<< (std::ostream& os) {
      A a();
      std::cout << a;
      return os;  
    }
};
