#include <stdio.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <fstream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define CHECK 1.0

__global__ void my_first_kernel(float *x, int niter) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    #pragma unroll
    for (int i=0; i<niter; i++) {
        x[tid] *= 2;
        x[tid] += 2;
        x[tid] /= 2;
        x[tid] -= 1;
    }
}

cudaDeviceProp getDeviceProperties() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        printf("No CUDA devices found\n");
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    return devProp;
}

void printDevProps(cudaDeviceProp devProp) {
    std::cout << "Device name: " << devProp.name << std::endl;
    std::cout << "Number of SMs: " << devProp.multiProcessorCount << std::endl;
    std::cout << "Maximum threads per block: " << devProp.maxThreadsPerBlock << std::endl;    
    std::cout << "Maximum blocks per grid: " << devProp.maxGridSize[0] << std::endl;
    std::cout << "Maximum threads per SM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum shared memory per block: " << devProp.sharedMemPerBlock << " B" << std::endl;
    std::cout << "Maximum shared memory per SM: " << devProp.sharedMemPerMultiprocessor << " B" << std::endl;
    std::cout << "Maximum global memory: " << devProp.totalGlobalMem << " B" << std::endl;
    std::cout << "Maximum blocks per SM: " << devProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Maximum threads per SM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
}

int main(int argc, const char **argv) {
    int delay = std::stoi(argv[1]);
    int niter = std::stoi(argv[2]);
    int testLength = std::stoi(argv[3]);
    std::string result_dir = argv[4];
    

    int dev = findCudaDevice(argc, (const char **) argv);
    if (dev == -1) return EXIT_FAILURE;
    cudaDeviceProp devProp = getDeviceProperties();
    // printDevProps(devProp);

    int nblocks  = devProp.maxBlocksPerMultiProcessor * devProp.multiProcessorCount;
    int nthreads = devProp.maxThreadsPerBlock;

    int nsize    = nblocks * nthreads;
    // std::cout << "nblocks: " << nblocks << std::endl;
    // std::cout << "nthreads: " << nthreads << std::endl;
    // std::cout << "nsize: " << nsize << std::endl;

    float *h_x, *d_x;
    h_x = (float *)malloc(nsize*sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

    // populate host array with 1
    for (int i=0; i<nsize; i++) h_x[i] = CHECK;

    // copy host array to device
    checkCudaErrors(cudaMemcpy(d_x,h_x,nsize*sizeof(float), cudaMemcpyHostToDevice));
    
    // Measurement begins
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    uint64_t timestamps[2*testLength+1];
    sleep(1);
    
    for (int i=0; i<testLength; i++) {
        timestamps[2*i] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        // cudaEventRecord(start); 

        my_first_kernel<<<nblocks,nthreads>>>(d_x, niter);
        getLastCudaError("my_first_kernel execution failed\n");
        
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "Elapsed time:    " << milliseconds << " ms" << std::endl;
        
        // Record the end time
        timestamps[2*i+1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        usleep(delay*1000);
        // std::cout << "Sleeping for:    " << delay << "      ms" << std::endl;
    }
    timestamps[2*testLength] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    sleep(1);
    
    // Write the timestamps to a file
    std::ofstream outfile;
    std::string filename = result_dir + "/timestamps.csv";
    outfile.open(filename);
    outfile << "timestamp" << std::endl;
    for (int i = 0; i < 2*testLength+1; i++) {
        outfile << timestamps[i] << std::endl;
        outfile << timestamps[i] << std::endl;
    }
    outfile.close();

    checkCudaErrors(cudaMemcpy(h_x,d_x,nsize*sizeof(float), cudaMemcpyDeviceToHost));

    // Check if the result is correct
    float sum = 0.0;
    for (int i=0; i<nsize; i++) sum += h_x[i];

    // raise a error if sum/nsize != CHECK
    if (sum/nsize != CHECK) {
        printf("Error: result is %f instead of %f\n", sum/nsize, CHECK);
        exit(EXIT_FAILURE);
    }
    

    // free memory 
    checkCudaErrors(cudaFree(d_x));
    free(h_x);

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();

    return 0;
}
