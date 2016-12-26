#ifndef UTIL_H
#define UTIL_H

#define rng(i, start, end) for(i = start; i < end; i++)
void check(int condition, const char *msg);

template<typename T> inline T* safe_malloc(int count) {
    T *array = (T *)malloc(count * sizeof(*array));
    check(!array, "safe_malloc failed"); 
    return array;
}
#endif
