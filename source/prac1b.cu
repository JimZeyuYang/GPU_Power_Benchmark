//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  x[tid] = (float) threadIdx.x;
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 

  // initialise card
  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  // execute kernel
  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out
  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 
  checkCudaErrors(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

  return 0;
}
