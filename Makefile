wd=$(shell basename $(shell pwd))
remote=ethanbro@tesla.ldc.upenn.edu
ssh_cmd=ssh -o ServerAliveInterval=60 $(remote)

SRCS=main.cpp util.cpp node.cpp
SRC_PATHS=$(addprefix $(wd)/,$(SRCS))

main: #$(SRC_PATHS)
	rsync -av *.cpp $(remote):~/$(wd)/
	#$(ssh_cmd) nvcc $(SRCS) -o $(WD)/main -lcublas -std=c++11
	#$(ssh_cmd) $(WD)/main

clean:
	rm *.o

