/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <signal.h>
#include <chrono>
#include <fstream>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* Main */
int main(int argc, char **argv) {
    int DELAY = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int testLength = std::stoi(argv[3]);

    cublasStatus_t status;
    float *h_A;     float *h_B;     float *h_C;
    float *d_A = 0; float *d_B = 0; float *d_C = 0;
    float alpha = 1.0f; float beta = 0.0f;
    int n2 = N * N;
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    int dev = findCudaDevice(argc, (const char **) argv);
    if (dev == -1) return EXIT_FAILURE;

    /* Initialize CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    h_B = (float *)malloc(n2 * sizeof(h_B[0]));
    if (h_B == 0) {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    h_C = (float *)malloc(n2 * sizeof(h_C[0]));
    if (h_C == 0) {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (int i = 0; i < n2; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    uint64_t timestamps[2*testLength];
    sleep(2);

    for (int i = 0; i < testLength; i++) {
        // Record the start time
        timestamps[2*i] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cudaEventRecord(start);
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! kernel execution error.\n");
            return EXIT_FAILURE;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

        // Record the end time
        timestamps[2*i+1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        usleep(DELAY*1000);
        std::cout << "Sleeping for:          " << DELAY << " ms" << std::endl;
    }

    sleep(1);

    // Write the timestamps to a file
    std::ofstream outfile;
    outfile.open("timestamps.csv");
    outfile << "timestamp" << std::endl;
    outfile << timestamps[0]-500000 << std::endl;
    for (int i = 0; i < 2*testLength; i++) {
        outfile << timestamps[i]-1000 << std::endl;
        outfile << timestamps[i]-1000 << std::endl;
    }
    outfile << timestamps[2*testLength-1]+500000 << std::endl;
    outfile.close();


    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
    free(h_A); free(h_B); free(h_C);

    if (cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_B) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_C) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Shutdown */
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
}
