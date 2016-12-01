SRCS:=main.cpp matrix.cu op.cu scan.cu
main: $(SRCS)  matrix.h op.h scan.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm main

rust: $(SRCS) matrix.h
	cd ..
	nvcc -c src/main.cpp -o out/main.o
	nvcc -c src/matrix.cu -o out/matrix.o
	nvcc -c src/op.cu -o out/op.o
	cd out
	ar curs libmatrix.a *.o

