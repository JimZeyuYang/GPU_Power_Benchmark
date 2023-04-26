
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas

NVCCFLAGS	:= -lineinfo -arch=sm_80 --ptxas-options=-v --use_fast_math

both:		benchmark

benchmark:	benchmark.cpp Makefile
		g++ benchmark.cpp -o benchmark $(INC) $(LIB)

clean:
		rm -f benchmark