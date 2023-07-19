/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Complex data type
typedef float2 Complex;

#define SIGNAL_SIZE 300000000
#define REPEAT (32)

int main(int argc, char **argv) { 
  findCudaDevice(argc, (const char **)argv); 

  // Allocate host memory for the signal
  Complex *h_signal = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
  // Initialize the memory for the signal
  for (unsigned long long int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
    h_signal[i].y = 0;
  }

  unsigned long long int mem_size = sizeof(Complex) * SIGNAL_SIZE;

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

  // CUFFT plan simple API
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1));



  for (int i = 0; i < 2; i++) cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal), reinterpret_cast<cufftComplex *>(d_signal), CUFFT_FORWARD);
  checkCudaErrors(cudaDeviceSynchronize());

  // printf("Transforming signal cufftExecC2C\n");

  uint64_t start_ts, end_ts;
  start_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  for (int i = 0; i < REPEAT; i++) cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal), reinterpret_cast<cufftComplex *>(d_signal), CUFFT_FORWARD);
  checkCudaErrors(cudaDeviceSynchronize());

  end_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();



  // printf("Done\n");
  // printf("Kernel Execution Time: %f ms\n", (end_ts - start_ts) / 1000.0 / REPEAT);
  // printf("Total runtime: %f ms\n", (end_ts - start_ts) / 1000.0);

  // Write the timestamps to a file
  std::ofstream outfile;
  outfile.open("timestamps.csv");
  outfile << "timestamp" << std::endl;
  outfile << start_ts << std::endl;
  outfile << end_ts << std::endl;
  outfile.close();

  // Copy device memory to host
  // Complex *h_fft_signal = h_signal;
  // checkCudaErrors(cudaMemcpy(h_fft_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  free(h_signal);
  checkCudaErrors(cudaFree(d_signal));

  exit(EXIT_SUCCESS);
}