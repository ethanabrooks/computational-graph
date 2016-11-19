wd=$(shell basename $(shell pwd))
remote=ethanbro@tesla.ldc.upenn.edu
ssh_cmd=ssh -o ServerAliveInterval=60 $(remote)

SRC_FILES=main.cpp util.cpp node.cpp
SRCS=$(addprefix $(wd)/,$(SRC_FILES))

main: $(SRC_FILES)
	rsync -av *.cpp $(remote):~/$(wd)/
	$(ssh_cmd) nvcc $(SRCS) -o $(WD)/main -lcublas -std=c++11
	$(ssh_cmd) $(WD)/main

clean:
	rm *.o

