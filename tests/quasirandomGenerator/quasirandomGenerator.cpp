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

// CUDA Runtime
#include <cuda_runtime.h>
#include <unistd.h>
// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include <chrono>
#include "quasirandomGenerator_common.h"

////////////////////////////////////////////////////////////////////////////////
// CPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);

extern "C" float getQuasirandomValue(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION], int i, int dim);

extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int p);

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initTableGPU(
    unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
extern "C" void quasirandomGeneratorGPU(float *d_Output, unsigned int seed,
                                        unsigned int N);
extern "C" void inverseCNDgpu(float *d_Output, unsigned int *d_Input,
                              unsigned int N);

const int N = 10485760;
#define REPEAT 7304
#define SHIFTS 1

int main(int argc, char **argv) {
  // Start logs
  // printf("%s Starting...\n\n", argv[0]);

  unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];

  float *h_OutputGPU, *d_Output;

  int dim, pos;
  double delta, ref, sumDelta, sumRef, L1norm, gpuTime;

  StopWatchInterface *hTimer = NULL;

  if (sizeof(INT64) != 8) {
    printf("sizeof(INT64) != 8\n");
    return 0;
  }

  // printf("Allocating GPU memory...\n");
  checkCudaErrors(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float)));

  // printf("Allocating CPU memory...\n");
  h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

  // printf("Initializing QRNG tables...\n\n");
  initQuasirandomGenerator(tableCPU);
  initTableGPU(tableCPU);

  // printf("Testing QRNG...\n\n");
  checkCudaErrors(cudaMemset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(float)));

  uint64_t time_array[SHIFTS*2];

  for (int i = 0; i < SHIFTS; i++) {
    time_array[i*2] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int j = 0; j < REPEAT/SHIFTS; j++)  quasirandomGeneratorGPU(d_Output, 0, N);
    cudaDeviceSynchronize();
    time_array[i*2+1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // sleep for 25 milliseconds
    usleep(25*1000);
  }


  // printf("Kernel Execution Time: %f ms\n", (end_ts - start_ts) / 1000.0 / REPEAT);
  // printf("Total runtim: %f ms\n", (end_ts - start_ts) / 1000.0);

  // Write the timestamps to a file
  std::ofstream outfile;
  outfile.open("timestamps.csv");
  outfile << "timestamp" << std::endl;
  for (int i = 0; i < SHIFTS*2; i++) {
    outfile << time_array[i] << std::endl;
  }
  outfile.close();

  
  
  exit(EXIT_SUCCESS);
}
