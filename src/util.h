#ifndef UTIL_H
#define UTIL_H

#define rng(i, start, end) for(i = start; i < end; i++)
void check(int condition, const char *msg);

template<typename T> T* safe_malloc(int count) {
    //printf("alloc\n");
    T *array = (T *)malloc(count * sizeof(T));
    check(!array, "safe_malloc failed"); 
    return array;
}

#endif
