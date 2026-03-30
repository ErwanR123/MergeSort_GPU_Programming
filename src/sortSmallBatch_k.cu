#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib> 
#include "timer.h"

using std::generate;

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Renamed macro to avoid naming conflicts with the function
#define CUDA_CHECK(error) (testCUDA(error, __FILE__ , __LINE__))


void sortSmallBatch_CPU(int* M_working_array, int d, int n_batch) {
    for (int batch = 0; batch < n_batch; batch++) {
        std::sort(M_working_array + (batch * d), M_working_array + ((batch + 1) * d));
    }
}

void verify_result(int* M, int d, int N){
    for (int batch = 0; batch < N ; batch++){
        for (int i = 0; i < d-1; i++){
            assert(M[i + d * batch] <= M[i+1 + d * batch]);
        }
    }
}

__global__ void sortSmallBatch_k(int* M_batch, int d, int N) {
    
    // Thread indices mapping
    int Qt = threadIdx.x / d;
    int tidx = threadIdx.x - (Qt * d);
    int gbx = Qt + blockIdx.x * (blockDim.x / d);

    // Double Buffering (Ping-Pong) in Shared Memory
    extern __shared__ int s_batch[];
    
    // Buffer A is the first half, Buffer B is the second half
    int* src = &s_batch[Qt * d];
    int* dst = &s_batch[blockDim.x + (Qt * d)]; 

    if (gbx >= N) return;

    // Load from Global Memory into the Source Buffer
    src[tidx] = M_batch[gbx * d + tidx];
    
    __syncthreads();

    // The Iterative Bottom-Up Merge Sort Loop
    for (int width = 1; width < d; width *= 2) {
        
        // Find the boundaries of the two chunks this specific thread is merging
        int left_start = (tidx / (2 * width)) * (2 * width);
        
        // Handle boundaries safely (in case d is not a perfect power of 2)
        int n_A = min(width, d - left_start);
        int n_B = min(width, max(0, d - left_start - width));

        int i = tidx - left_start; // Local thread ID for this specific merge pair

        if (n_B == 0) {
            // If there is no right array to merge with, just carry the data over
            dst[tidx] = src[tidx];
        } else {
            // Pointers to the two specific sub-arrays we are merging
            int* A = src + left_start;
            int* B = src + left_start + width;

            int Kx, Ky, Px, Py;
            if (i > n_A) {
                Kx = i - n_A;  Ky = n_A;
                Px = n_A;      Py = i - n_A;
            } else {
                Kx = 0;        Ky = i;
                Px = i;        Py = 0;
            }

            while (true) {
                int offset = abs(Ky - Py) / 2;
                int Qx = Kx + offset;
                int Qy = Ky - offset;

                if (Qy >= 0 && Qx <= n_B && (Qy == n_A || Qx == 0 || A[Qy] > B[Qx - 1])) {
                    if (Qx == n_B || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                        
                        // Write to the DESTINATION buffer
                        if (Qy < n_A && (Qx == n_B || A[Qy] <= B[Qx])){
                            dst[tidx] = A[Qy];
                        } else {
                            dst[tidx] = B[Qx];
                        }
                        break; // Found it! Exit the while loop.
                    } else {
                        Kx = Qx + 1; Ky = Qy - 1;
                    }
                } else {
                    Px = Qx - 1; Py = Qy + 1;
                }
            }
        }

        __syncthreads();

        // Swap the pointers for the next pass!
        int* temp = src;
        src = dst;
        dst = temp;
    }

    // Write back to Global Memory
    M_batch[gbx * d + tidx] = src[tidx];
}

float bench_sort_with_timer(int* d_M, int* h_M_in, int d, int N, int totalElems, Timer& Tim) {
    
    int tpb = (1024 / d) * d;
    int groupesParBloc = tpb / d;
    int numBlocks = (N + groupesParBloc - 1) / groupesParBloc;
    
    // CRITICAL: Double Shared Memory for the Ping-Pong buffers
    int sharedMemBytes = 2 * tpb * sizeof(int); 

    // 1. Refresh the GPU memory with completely unsorted data 
    // We do this OFF the clock so we only measure computational sorting time!
    CUDA_CHECK(cudaMemcpy(d_M, h_M_in, totalElems * sizeof(int), cudaMemcpyHostToDevice));

    // 2. Start timing the actual sorting process
    Tim.reset();
    Tim.start();
    
    sortSmallBatch_k<<<numBlocks, tpb, sharedMemBytes>>>(d_M, d, N);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    Tim.add();

    // Return milliseconds
    return Tim.getsum() * 1000.0f; 
}

int main() {
    Timer Tim;
    int n_batch, d;

    printf("\n========== Interactive GPU Sort Test ==========\n");
    printf("Number of batches to sort: ");
    std::cin >> n_batch;
    printf("Number of elements per batch 'd' (<= 1024): ");
    std::cin >> d;

    assert(d <= 1024 && "Error: d cannot exceed 1024 for this shared memory kernel!");
    
    int total_elements = n_batch * d;

    // Allocate Host Memory
    int* h_M_in  = new int[total_elements]; // Unsorted raw data
    int* h_M_cpu = new int[total_elements]; // CPU sorted results
    int* h_M_gpu = new int[total_elements]; // GPU sorted results

    // Generate completely random, unsorted arrays
    std::generate(h_M_in, h_M_in + total_elements, []() { return rand() % 5000; });
    
    // CPU SORT
    std::copy(h_M_in, h_M_in + total_elements, h_M_cpu);

    Tim.reset(); Tim.start();
    sortSmallBatch_CPU(h_M_cpu, d, n_batch);
    Tim.add();
    float cpu_time = Tim.getsum();
    verify_result(h_M_cpu, d, n_batch);
    printf("\nCPU Sort completed in %f s.\n", cpu_time);

    // GPU SORT SETUP
    int *d_M;
    CUDA_CHECK(cudaMalloc((void**)&d_M, total_elements * sizeof(int)));
    
    int threadsPerBlock  = (1024 / d) * d;
    int groupesPerBlock  = threadsPerBlock / d;
    int numBlocks        = (n_batch + groupesPerBlock - 1) / groupesPerBlock;

    // Multiply by 2 because we need Buffer A and Buffer B for Ping-Pong!
    int sharedMemBytes   = 2 * threadsPerBlock * sizeof(int);

    // GPU WARMUP & TIMING
    // Copy unsorted data to Device
    CUDA_CHECK(cudaMemcpy(d_M, h_M_in, total_elements * sizeof(int), cudaMemcpyHostToDevice));
    
    // Warm-up
    sortSmallBatch_k <<<numBlocks, threadsPerBlock, sharedMemBytes>>> (d_M, d, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-copy fresh unsorted data for the actual timed run
    CUDA_CHECK(cudaMemcpy(d_M, h_M_in, total_elements * sizeof(int), cudaMemcpyHostToDevice));

    Tim.reset(); Tim.start();
    sortSmallBatch_k <<<numBlocks, threadsPerBlock, sharedMemBytes>>> (d_M, d, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Tim.add();
    float gpu_time = Tim.getsum();

    // Copy results back to Host
    CUDA_CHECK(cudaMemcpy(h_M_gpu, d_M, total_elements * sizeof(int), cudaMemcpyDeviceToHost));

    verify_result(h_M_gpu, d, n_batch);
    printf("GPU Sort completed in %f s.\n", gpu_time);

    // Free memory
    CUDA_CHECK(cudaFree(d_M));
    delete[] h_M_in;
    delete[] h_M_cpu;
    delete[] h_M_gpu;

    // CONSTANT WORKLOAD BENCHMARK (Full Sort with FRESH DATA per 'd')
    int totalElems = 1 << 24;  // ~16.7 million elements

    printf("\n========== Constant Workload Benchmark (Full Sort) ==========\n");
    printf("Allocating %d elements...\n", totalElems);

    // 1. Allocate Memory ONCE (Size never changes)
    int* h_M_in_test = new int[totalElems];
    int* d_M_test;
    CUDA_CHECK(cudaMalloc(&d_M_test, totalElems * sizeof(int)));

    printf("\n=== Execution Time vs Array Size (d) ===\n");
    printf("%-8s %-12s %-10s %-14s %-10s\n", "d", "N", "tpb", "groupes/bloc", "temps(ms)");
    printf("----------------------------------------------------------------\n");

    int ds[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int nd   = sizeof(ds) / sizeof(ds[0]);

    // WARMUP RUN
    std::generate(h_M_in_test, h_M_in_test + totalElems, []() { return rand() % 10000; });
    bench_sort_with_timer(d_M_test, h_M_in_test, 1024, totalElems / 1024, totalElems, Tim);

    // 2. Loop through all values of 'd'
    for (int k = 0; k < nd; k++) {
        int d_bench = ds[k];
        int N_bench = totalElems / d_bench;  

        // =================================================================
        // THE FIX: Generate completely FRESH random data for this exact 'd'
        // =================================================================
        std::generate(h_M_in_test, h_M_in_test + totalElems, []() { return rand() % 10000; });

        int tpb_bench = (1024 / d_bench) * d_bench;
        int groupesParBloc_bench = tpb_bench / d_bench;

        // Run the benchmark (copies the fresh data to GPU and times it)
        float ms = bench_sort_with_timer(d_M_test, h_M_in_test, d_bench, N_bench, totalElems, Tim);
        
        printf("%-8d %-12d %-10d %-14d %.3f\n", d_bench, N_bench, tpb_bench, groupesParBloc_bench, ms);
    }
    printf("----------------------------------------------------------------\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_M_test)); 
    delete[] h_M_in_test;
}