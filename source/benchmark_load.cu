#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

#include <stdio.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "PMD.h"

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
    if (devCount == 0)  printf("No CUDA devices found\n");

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


void start_PMD(int& serial_port, uint64_t& PMD_start, pthread_t& serialThread, ThreadArgs* args, std::string result_dir) {
    serial_port = open_serial_port();
    change_baud_rate(serial_port, 921600);
    if (!handshake(serial_port)) {
        std::cout << "Error: handshake failed" << std::endl;
        exit(-1);
    }

    args = new ThreadArgs;
    args->serial_port = serial_port;
    args->file_path = result_dir;

    config_cont_tx(serial_port, true);
    PMD_start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    
    int rc = pthread_create(&serialThread, NULL, logSerialPort, args);
    if (rc) {
        std::cout << "Error: unable to create thread, " << rc << std::endl;
        exit(-1);
    }
}

void stop_PMD(int& serial_port, pthread_t& serialThread, ThreadArgs*& args, std::string result_dir, uint64_t PMD_start) {
    config_cont_tx(serial_port, false);
    pthread_join(serialThread, NULL);
    delete args;
    change_baud_rate(serial_port, 115200);
    close(serial_port);

    std::ofstream outfile;
    std::string filename = result_dir + "/PMD_start_ts.txt";
    outfile.open(filename);
    outfile << PMD_start << std::endl;
    outfile.close();
} 


int main(int argc, const char **argv) {
    int experiment = std::stoi(argv[1]);
    std::string config = argv[2];
    std::string result_dir = argv[3];
    bool NVML = std::stoi(argv[4]);
    bool PMD = std::stoi(argv[5]);
    int gpu_id = std::stoi(argv[6]);
    
    std::stringstream ss(config);  std::string token;

    cudaSetDevice(gpu_id);

    // initialize GPU and get device properties
    int dev = findCudaDevice(argc, (const char **) argv);
    if (dev == -1) return EXIT_FAILURE;
    cudaDeviceProp devProp = getDeviceProperties();
    // printDevProps(devProp);


    if (experiment == 1) {  // Experiment 1: Steady state and transient response / avg window analysis
        // Read the config variables
        getline(ss, token, ',');  int delay      = std::stoi(token);
        getline(ss, token, ',');  int niter      = std::stoi(token);
        getline(ss, token, ',');  int testLength = std::stoi(token);
        getline(ss, token, ',');  int percent    = std::stoi(token);

        float *h_x, *d_x;
        int nblocks, nthreads, nsize;
        // begin experiment
        // nblocks = devProp.maxBlocksPerMultiProcessor * devProp.multiProcessorCount;
        nblocks = devProp.multiProcessorCount * percent / 100;
        if (nblocks < 1) nblocks = 1;
        nthreads = devProp.maxThreadsPerBlock;
        nsize    = nblocks * nthreads;

        h_x = (float *)malloc(nsize*sizeof(float));
        checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

        // populate host array with 1
        for (int i=0; i<nsize; i++) h_x[i] = CHECK;

        // copy host array to device
        checkCudaErrors(cudaMemcpy(d_x,h_x,nsize*sizeof(float), cudaMemcpyHostToDevice));
        
        // Measurement begins
        //cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        
        // Record data on PMD if enabled
        int serial_port = 0; uint64_t PMD_start = 0; pthread_t serialThread; ThreadArgs* args = nullptr;
        if (PMD) start_PMD(serial_port, PMD_start, serialThread, args, result_dir);


        uint64_t timestamps[2*testLength+1];
        sleep(1);
        
        for (int i=0; i<testLength; i++) {
            timestamps[2*i] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            // cudaEventRecord(start); 

            my_first_kernel<<<nblocks,nthreads>>>(d_x, niter);
            // getLastCudaError("my_first_kernel execution failed\n");
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("my_first_kernel execution failed. CUDA error: %s\n", cudaGetErrorString(err));
            }
            
            // cudaEventRecord(stop);  cudaEventSynchronize(stop);
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

        if (PMD) stop_PMD(serial_port, serialThread, args, result_dir, PMD_start);

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
        
        // Copy the result from device to host
        checkCudaErrors(cudaMemcpy(h_x,d_x,nsize*sizeof(float), cudaMemcpyDeviceToHost));
        // Check if the result is correct
        float sum = 0.0;
        for (int i=0; i<nsize; i++) sum += h_x[i];

        // raise a error if sum/nsize != CHECK
        if (sum/nsize != CHECK) {
            printf("Warning: result is %f instead of %f\n", sum/nsize, CHECK);
            // exit(EXIT_FAILURE);
        }
        
        // free memory 
        checkCudaErrors(cudaFree(d_x));  free(h_x);


    } else if (experiment == 2) {  // Run a executable and take the measurements
        getline(ss, token, ','); std::string path = token;
        getline(ss, token, ','); std::string executable = token;

        std::string cmd = "(cd " + path + " && " + executable + ")";
        // std::cout << "Running the executable: " << cmd << std::endl;

        int serial_port = 0; uint64_t PMD_start = 0; pthread_t serialThread; ThreadArgs* args = nullptr;
        if (PMD) start_PMD(serial_port, PMD_start, serialThread, args, result_dir);
        system(cmd.c_str());
        if (PMD) stop_PMD(serial_port, serialThread, args, result_dir, PMD_start);

        std::string mv_data = "mv " + path + "/timestamps.csv " + result_dir;
        // std::cout << "Moving the timestamps file: " << mv_data << std::endl;
        system(mv_data.c_str());

    } else {
            std::cout << "Invalid experiment number" << std::endl;
            exit(EXIT_FAILURE);
    }




    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();

    return 0;
}
