SRCS:=main.cpp util.cpp node.cpp
main: $(SRCS) util.h matrix.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm *.o

