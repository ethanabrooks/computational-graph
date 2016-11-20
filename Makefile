SRCS:=main.cpp util.cpp
main: $(SRCS) util.hpp graph.hpp matrix.hpp cout-test.hpp
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm *.o

