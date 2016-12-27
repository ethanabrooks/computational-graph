SRCS:=main.cpp matrix.cu ops.cu util.cu
main: $(SRCS)  matrix.h ops.h util.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm -f main
