SRCS:=cpu/main.cpp cpu/matrix.cpp cpu/ops.cpp cpu/util.cpp
main: $(SRCS)  header/matrix.h header/ops.h header/util.h
	cc -I header/ $(SRCS) -o main -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm -f main
