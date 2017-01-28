#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <iostream> 
#include <time.h>
#include "matrix.h" 
#include "ops.h" 
#include "util.h" 

#define L 2
#define M 3
#define N 2

// globals
float *input;
float *devPtrInput;
float *weights;
float *devPtrWeights;

int main (void) { 

  // set values of input matrix
  float input_vals [] = {
    1, 2, 
    3, 4
  };

  // set values of weights matrix
  float weights_vals [] = {
    1, 2, 
    3, 4,
    5, 6
  };

  int i, j;
  float *array;
  Matrix m1;
  Matrix m2 = {.height=2, .width=2};
  Matrix result;

  array = (float *)malloc (M * N * sizeof (float)); 
  check(!array, "host memory allocation failed"); 
  init_matrix(&m1, weights_vals, 3, 2);

  //struct car c = {.year=1923, .make="Nash", .model="48 Sports Touring Car"};
  alloc_matrix(&m2);
  fill_matrix(&m2, 4);
  print_matrix(&m1);
  printf("\n");
  print_matrix(&m2);
  return EXIT_SUCCESS;
}
