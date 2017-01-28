#ifndef UTIL_H
#define UTIL_H

#define rng(i, start, end) for(i = start; i < end; i++)
void check(int condition, const char *msg);

template<typename T> T* safe_malloc(int count) {
    //printf("C test start\n");
    float *test = (float*)malloc(count * sizeof(float));
    //printf("C test end\n");
    //printf("C bef\n");
    //printf("count: %d\n", count);
    //printf("sizeof(T): %d\n", sizeof(T));
    //printf("sizeof(float): %d\n", sizeof(float));
    //T *array = (T *)malloc(count * sizeof(float));
    T *array = (T*)test;
    //printf("C aft\n");
    check(!array, "safe_malloc failed"); 
    return array;
}

#endif
