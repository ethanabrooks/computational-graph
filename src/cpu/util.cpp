#include <stdio.h> 
#include <stdlib.h> 
#include "util.h" 

void check(int condition, const char *msg) {
  if (condition) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
  }
}
