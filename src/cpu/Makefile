SRCS:=main.cpp matrix.cu ops.cu util.cu
main: $(SRCS)  matrix.h ops.h util.h
	nvcc $(SRCS) -o main -lcublas -std=c++11
	./main

clean:
	rm -f *.o
	rm -f *.out
	rm -f main

rust: $(SRCS) matrix.h
	cd ..
	nvcc -c src/main.cpp -o out/main.o
	nvcc -c src/matrix.cu -o out/matrix.o
	nvcc -c src/op.cu -o out/op.o
	cd out
	ar curs libmatrix.a *.o

rust_clean: 
	rm -rf ../target/debug/build/lstm-53f3b05885762f6a
