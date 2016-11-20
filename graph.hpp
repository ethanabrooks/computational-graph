#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <cuda_runtime.h> 
#include <boost/optional.hpp>
#include "cublas_v2.h" 
#include "util.hpp" 
#include "constant.hpp" 
#endif

class AddNode;

class Graph {
  private:
    virtual std::ostream& write(std::ostream& os) = 0;

  public:
    AddNode operator+(Graph &g);
    virtual Constant& eval() = 0;

  friend std::ostream& operator<< (std::ostream& stream, Graph &graph) { 
    graph.write(stream);
  }
};

class FloatNode: public Graph {
  private: 
    float value;
    std::ostream& write(std::ostream& os) {
      os << value;
      return os;
    };

  public:
    FloatNode(float x) { value = x; }
    Constant& eval() {
      return Float(value);
    }
};

class AddNode: public Graph {
  private:
    Graph *left;
    Graph *right;
  public:
    AddNode(Graph *left, Graph *right) {
      this->left = left;
      this->right = right;
    }

    Constant& eval() { 
      return left->eval() + right->eval(); 
    }

    std::ostream& write(std::ostream& os) {
      std::cout << "add(" << *left << ", " << *right << ")";
      return os;
    }
};

AddNode Graph::operator+(Graph &g) {
  return AddNode(this, &g);
};

