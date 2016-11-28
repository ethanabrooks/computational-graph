SRCS:=main.cpp matrix.cu op.cu
main: $(SRCS)  matrix.h op.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm *.o
	rm main

rust: $(SRCS) matrix.h
	cd ..
	nvcc src/main.cpp src/matrix.cu src/op.cu -c -o out/main.o
	cd out
	ar curs libmatrix.a main.o

