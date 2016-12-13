#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "matrix.h"
#include "util.h"

#define IDx_T ((IDx) % (width)) * (width) + ((IDx) / (width))
#define IDX(i, j, lda) ((j) * (lda) + (i))
#define UN_MAP(name, f_body) \
  float f_map_ ## name(float x) { return f_body; } \
  void map_ ## name(const Matrix *m, Matrix *result) { \
    CHECK_EQUAL(m->height, result->height); \
    CHECK_EQUAL(m->width, result->width); \
    int i; \
    rng(i, 0, size(m)) { \
      result->array[i] = f_map_ ## name(m->array[i]); \
    } \
  }

#define BIN_BROADCAST(name, op) \
  float f_broadcast_ ## name(float val1, float val2) { \
    return val1 op val2; \
  } \
  void broadcast_ ## name(float val, const Matrix *m, Matrix *result) { \
    CHECK_EQUAL(m->height, result->height); \
    CHECK_EQUAL(m->width, result->width); \
    int i; \
    rng(i, 0, size(m)) { \
      result->array[i] = f_broadcast_ ## name(val, m->array[i]); \
    } \
  } \

#define BIN_BROADCAST_REV(name, op) \
  float f_broadcast_rev_ ## name(float val1, float val2) { \
    return val1 op val2; \
  } \
  void broadcast_rev_ ## name(float val, const Matrix *m, Matrix *result) { \
    CHECK_EQUAL(m->height, result->height); \
    CHECK_EQUAL(m->width, result->width); \
    int i; \
    rng(i, 0, size(m)) { \
      result->array[i] = f_broadcast_rev_ ## name(m->array[i], val); \
    } \
  } \

#define BIN_ELEMWISE(name, op) \
  float f_elemwise_ ## name(float val1, float val2) { \
    return val1 op val2; \
  } \
  void elemwise_ ## name(const Matrix *m1, const Matrix *m2, Matrix *result) { \
    check_all_eq(m1, m2, result); \
    int i; \
    rng(i, 0, size(m1)) { \
      result->array[i] = f_elemwise_ ## name(m1->array[i], m2->array[i]); \
    } \
  } \

#define CHECK_EQUAL(side1, side2) \
  check(side1 != side2,  #side1 " must equal " #side2)

void check_all_eq(const Matrix *m1, const Matrix *m2, const Matrix *result) {
  CHECK_EQUAL(m1->height, m2->height);
  CHECK_EQUAL(m1->width, m2->width);
  CHECK_EQUAL(m1->height, result->height);
  CHECK_EQUAL(m1->width, result->width);
}

extern "C" {
  UN_MAP(neg, -x) // map_neg
  UN_MAP(sq, x * x) // map_sq
  UN_MAP(abs, x < 0 ? -x : x) // map_aps
  UN_MAP(signum, x < 0 ? -1 : 1) // map_signum
  UN_MAP(sigmoid, 1.0f / (1.0f + expf(-x))) // map_sigmoid
  UN_MAP(tanh, tanh(x)) // map_tanh
  UN_MAP(one_minus, 1.0f - x) // map_one_minus

  BIN_ELEMWISE(mul, *) // elemwise_mult
  BIN_ELEMWISE(add, +) // elemwise_add
  BIN_ELEMWISE(sub, -) // elemwise_sub

  BIN_BROADCAST(mul, *) // broadcast_mult
  BIN_BROADCAST(add, +) // broadcast_add
  BIN_BROADCAST(sub, -) // broadcast_sub

  BIN_BROADCAST_REV(sub, -) // broadcast_sub_rev
  BIN_BROADCAST_REV(mul, *) // broadcast_mul_rev
  BIN_BROADCAST_REV(add, +) // broadcast_add_rev

  void gemm(const Matrix *m1, bool trans1,
            const Matrix *m2, bool trans2,
            Matrix *result) {

    int height1 = trans1 ? m1->width : m1->height;
    int inner_dim = trans1 ? m1->height : m1->width;
    int height2 = trans2 ? m1->width : m1->height;
    int width2 = trans2 ? m1->height : m1->width;

    CHECK_EQUAL(height1, result->height);
    CHECK_EQUAL(inner_dim, height2);
    CHECK_EQUAL(width2, result->width);

    bzero(result->array, size(result) * sizeof(float));

    int i, j, k;
    rng(i, 0, height1) {
      rng(j, 0, width2) {
        rng(k, 0, inner_dim) {
          int idx1 = trans1 ? IDX(k, i, m1->height) : IDX(i, k, m1->height);
          int m1_ik = m1->array[idx1];
          int idx2 = trans2 ? IDX(j, k, m2->height) : IDX(k, j, m2->height);
          int m2_kj = m2->array[idx2];
          result->array[IDX(i, j, result->width)] += m1_ik * m2_kj;
        }
      }
    }

  }

  bool reduce_equal(const Matrix *m, float x) {
    int i;
    rng(i, 0, size(m)) {
      if (m->array[i] != x) {
        return false;
      }
    }
    return true;
  }

  float reduce_sum(const Matrix *m) {
    float sum = 0;
    int i;
    rng(i, 0, size(m)) {
      sum += m->array[i];
    }
    return sum;
  }
}

