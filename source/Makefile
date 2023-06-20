INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas

NVCCFLAGS	:= -lineinfo --ptxas-options=-v --use_fast_math
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11

BINDIR	:= /tmp
OBJS := PMD.o

all:	$(BINDIR)/benchmark_load

.PHONY: clean

$(BINDIR)/benchmark_load:	benchmark_load.cu $(OBJS) Makefile
		nvcc benchmark_load.cu $(OBJS) -o $(BINDIR)/benchmark_load $(INC) $(NVCCFLAGS) $(LIB)

PMD.o: PMD.cpp PMD.h
		g++ $(CXXFLAGS) -c PMD.cpp

clean:
		rm -f $(BINDIR)/benchmark_load $(OBJS)
