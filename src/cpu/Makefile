SRCS:=main.cpp matrix.cpp ops.cpp util.cpp
main: $(SRCS)  matrix.h ops.h util.h
	c++ $(SRCS) -o main -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm -f main
