SRCS:=main.cpp matrix.cu op.cu
main: $(SRCS)  matrix.h op.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm *.o
	rm main

rust: $(SRCS) matrix.h
	nvcc -c main.o
	cargo run

