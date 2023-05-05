
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas

NVCCFLAGS	:= -lineinfo --ptxas-options=-v --use_fast_math

BINDIR	:= ../bin

all:	$(BINDIR)/benchmark_load

$(BINDIR)/benchmark_load:	benchmark_load.cu Makefile
		nvcc benchmark_load.cu -o $(BINDIR)/benchmark_load $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f $(BINDIR)/benchmark_load