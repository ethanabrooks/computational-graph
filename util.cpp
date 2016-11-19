#include <stdio.h> 
#include <stdlib.h> 
#include "util.h" 

int idx2c(int i, int j, int width) {
 return j * width + i;
}


void check(bool condition, const char *msg) {
  if (condition) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
  }
}

