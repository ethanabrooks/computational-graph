SRCS:=cpu/main.cpp cpu/matrix.cpp cpu/ops.cpp cpu/util.cpp
main: $(SRCS)  cpu/matrix.h cpu/ops.h cpu/util.h
	cc $(SRCS) -o main -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm -f main
