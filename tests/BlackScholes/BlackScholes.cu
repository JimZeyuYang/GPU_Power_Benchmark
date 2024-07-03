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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization
#include <chrono>
#include <unistd.h>
////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                                float *h_StockPrice, float *h_OptionStrike,
                                float *h_OptionYears, float Riskfree,
                                float Volatility, int optN);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 100000000;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  // printf("[%s] - Starting...\n", argv[0]);

  //'h_' prefix - CPU (host) memory space
  float
      // Results calculated by CPU for reference
      *h_CallResultCPU,
      *h_PutResultCPU,
      // CPU copy of GPU results
      *h_CallResultGPU, *h_PutResultGPU,
      // CPU instance of input data
      *h_StockPrice, *h_OptionStrike, *h_OptionYears;

  //'d_' prefix - GPU (device) memory space
  float
      // Results calculated by GPU
      *d_CallResult,
      *d_PutResult,
      // GPU instance of input data
      *d_StockPrice, *d_OptionStrike, *d_OptionYears;

  double gpuTime;

  StopWatchInterface *hTimer = NULL;
  int i;

  findCudaDevice(argc, (const char **)argv);

  // Variables to store the values of flags
  int REPEAT = 1;
  int SHIFTS = 1;

  // Parsing command line arguments
  for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-r") == 0) {
          if (i + 1 < argc) { // Ensure there is an argument to consume
              REPEAT = std::atoi(argv[++i]); // Increment 'i' to skip next argument
          } else {
              std::cerr << "Error: Option '-r' requires an integer value." << std::endl;
              return EXIT_FAILURE;
          }
      } else if (strcmp(argv[i], "-s") == 0) {
          if (i + 1 < argc) { // Ensure there is an argument to consume
              SHIFTS = std::atoi(argv[++i]); // Increment 'i' to skip next argument
          } else {
              std::cerr << "Error: Option '-s' requires an integer value." << std::endl;
              return EXIT_FAILURE;
          }
      }
  }

  sdkCreateTimer(&hTimer);

  // printf("Initializing data...\n");
  // printf("...allocating CPU memory for options.\n");
  h_CallResultCPU = (float *)malloc(OPT_SZ);
  h_PutResultCPU = (float *)malloc(OPT_SZ);
  h_CallResultGPU = (float *)malloc(OPT_SZ);
  h_PutResultGPU = (float *)malloc(OPT_SZ);
  h_StockPrice = (float *)malloc(OPT_SZ);
  h_OptionStrike = (float *)malloc(OPT_SZ);
  h_OptionYears = (float *)malloc(OPT_SZ);

  // printf("...allocating GPU memory for options.\n");
  checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

  // printf("...generating input data in CPU mem.\n");
  srand(5347);

  // Generate options set
  for (i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }

  // printf("...copying input data to GPU mem.\n");
  // Copy options data to GPU memory for further processing
  checkCudaErrors(
      cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice));
  // printf("Data init done.\n\n");

  // printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
  for (i = 0; i < 10; i++) {
    BlackScholesGPU<<<DIV_UP((OPT_N / 2), 128), 128 /*480, 128*/>>>(
        (float2 *)d_CallResult, (float2 *)d_PutResult, (float2 *)d_StockPrice,
        (float2 *)d_OptionStrike, (float2 *)d_OptionYears, RISKFREE, VOLATILITY,
        OPT_N);
    getLastCudaError("BlackScholesGPU() execution failed\n");
  }
  checkCudaErrors(cudaDeviceSynchronize());




  uint64_t time_array[SHIFTS*2];

  for (int i = 0; i < SHIFTS; i++) {
    time_array[i*2] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int j = 0; j < REPEAT/SHIFTS; j++) {
      BlackScholesGPU<<<DIV_UP((OPT_N / 2), 128), 128 /*480, 128*/>>>(
          (float2 *)d_CallResult, (float2 *)d_PutResult, (float2 *)d_StockPrice,
          (float2 *)d_OptionStrike, (float2 *)d_OptionYears, RISKFREE, VOLATILITY,
          OPT_N);
      getLastCudaError("BlackScholesGPU() execution failed\n");
    }
    cudaDeviceSynchronize();
    time_array[i*2+1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // sleep for 25 milliseconds
    usleep(25*1000);
  }


  // Write the timestamps to a file
  std::ofstream outfile;
  outfile.open("timestamps.csv");
  outfile << "timestamp" << std::endl;
  for (int i = 0; i < SHIFTS*2; i++) {
    outfile << time_array[i] << std::endl;
  }
  outfile.close();

  // printf("Kernel Execution Time: %f ms\n", (end_ts - start_ts) / 1000.0 / NUM_ITERATIONS);
  // printf("Total runtim: %f ms\n", (end_ts - start_ts) / 1000.0);




  exit(EXIT_SUCCESS);
}
