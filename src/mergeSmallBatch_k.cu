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

void mergeSmallBatch_CPU(int* A, int* B,
                         int n_A, int n_B,
                         int N, int* M){

    for (int batch = 0; batch < N; batch++){
        int* myA = A + batch * n_A;
        int* myB = B + batch * n_B;
        int* myM = M + batch * (n_A + n_B);

        int i = 0;
        int j = 0;

        while (i+j < n_A + n_B)
        {
            if (i >= n_A){
                myM[i+j] = myB[j];
                j += 1;
            }
            else if (j >= n_B)
            {
                myM[i+j] = myA[i];
                i += 1;
            }
            else if (myA[i] <= myB[j])
            {
                myM[i+j] = myA[i];
                i += 1;
            }
            else
            {
                myM[i+j] = myB[j];
                j += 1;
            }
        }    
    }
}

__global__ void mergeSmallBatch_k(int* A, int* B, int* M,
                                   int n_A, int n_B, int N) {
    int d = n_A + n_B;
    
    // Local fusion id among same block 
    int Qt   = threadIdx.x / d;
    // Local thread id among same fusion
    int tidx = threadIdx.x - (Qt * d);
    // Global fusion id among all the pairs
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);

    if (gbx >= N) return;

    // Pointeurs vers la paire gbx dans les tableaux batch
    int* myA = A + gbx * n_A;
    int* myB = B + gbx * n_B;
    int* myM = M + gbx * d;

    // Same merge path and same logic as mergeSmall_k
    int i = tidx;

    int Kx, Ky, Px, Py;
    if (i > n_A) {
        Kx = i - n_A;  
        Ky = n_A;

        Px = n_A;
        Py = i - n_A;

    } else {
        Kx = 0;
        Ky = i;

        Px = i;
        Py = 0;
    }

    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= n_B &&
            (Qy == n_A || Qx == 0 || myA[Qy] > myB[Qx - 1])) {

            if (Qx == n_B || Qy == 0 || myA[Qy - 1] <= myB[Qx]) {

                if (Qy < n_A && (Qx == n_B || myA[Qy] <= myB[Qx])) {
                    myM[i] = myA[Qy];
                } else {
                    myM[i] = myB[Qx];
                }
                return;

            } else {
                Kx = Qx + 1;
                Ky = Qy - 1;
            }

        }   else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

// Here we try to use shared memory to reduce the compute time
// In reality we obtain worse result as we have bank acces issues
__global__ void mergeSmallBatch_k_optimized(int* A_batch, int* B_batch,
                                            int* M_batch, int n_A, 
                                            int n_B, int N) {
    
    int d = n_A + n_B; 

    int Qt = threadIdx.x / d;               
    int tidx = threadIdx.x - (Qt * d);      
    int gbx = Qt + blockIdx.x * (blockDim.x / d);

    // Dynamically allocate Shared Memory for the ENTIRE block
    extern __shared__ int s_batch[];

    if (gbx < N) {
        
        int shared_offset = Qt * d; // Where does this pair's chunk start?
        
        if (tidx < n_A) {
            // Load from A
            s_batch[shared_offset + tidx] = A_batch[(gbx * n_A) + tidx];
        } else {
            // Load from B
            s_batch[shared_offset + tidx] = B_batch[(gbx * n_B) + (tidx - n_A)];
        }
    }

    __syncthreads();

    if (gbx >= N) return;

    // This points exactly to where this thread's specific arrays live in shared memory
    int* s_A = &s_batch[Qt * d];
    int* s_B = &s_batch[(Qt * d) + n_A];

    int i = tidx; 

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

        if (Qy >= 0 && Qx <= n_B &&
            (Qy == n_A || Qx == 0 || s_A[Qy] > s_B[Qx - 1])) {

            if (Qx == n_B || Qy == 0 || s_A[Qy - 1] <= s_B[Qx]) {
                
                if (Qy < n_A && (Qx == n_B || s_A[Qy] <= s_B[Qx])){
                    M_batch[(gbx * d) + i] = s_A[Qy];
                } else {
                    M_batch[(gbx * d) + i] = s_B[Qx];
                }
                return;
            } else {
                Kx = Qx + 1; Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1; Py = Qy + 1;
        }
    }
}

void verify_result(int* M, int d, int N){
    for (int batch = 0; batch < N ; batch++){
        for (int i = 0; i < d-1; i++){
            assert(M[i + d * batch] <= M[i+1 + d * batch]);
        }
    }
}

float bench_with_timer(int* d_A, int* d_B, int* d_M, int n_A, int n_B, int d, int N, Timer& Tim) {
    int tpb = (1024 / d) * d;
    int groupesParBloc = tpb / d;
    int numBlocks = (N + groupesParBloc - 1) / groupesParBloc;

    Tim.reset();
    Tim.start();
    
    // Launch the global memory kernel
    mergeSmallBatch_k<<<numBlocks, tpb>>>(d_A, d_B, d_M, n_A, n_B, N);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    Tim.add();

    // Tim.getsum() returns seconds, multiply by 1000 for milliseconds
    return Tim.getsum() * 1000.0f; 
}


int main() {

    Timer Tim;
    
    // =====================================================================
    // PART 1: PERSONALIZED INTERACTIVE TEST
    // =====================================================================
    int n_A, n_B, n_batch, d;

    printf("\n========== Hand made test ==========\n");
    printf("Number of batchs : ");
    std::cin >> n_batch;
    printf("Number of elements in each array A (<1024): ");
    std::cin >> n_A;
    printf("Number of elements in each array B (<1024 - n_A): ");
    std::cin >> n_B;

    d = n_A + n_B;
    assert(d <= 1024);
    
    int* h_A = new int[n_batch * n_A];
    int* h_B = new int[n_batch * n_B];
    int* h_M = new int[n_batch * d];

    // Generate random arrays
    std::generate(h_A, h_A + (n_batch * n_A), []() { return rand() % 1000; });
    std::generate(h_B, h_B + (n_batch * n_B), []() { return rand() % 1000; });

    // Sort the arrays per group before merging them
    for (int batch = 0; batch < n_batch; batch++){
        std::sort(h_A + (batch * n_A), h_A + (batch + 1) * n_A );
        std::sort(h_B + (batch * n_B), h_B + (batch + 1) * n_B );
    }
    
    // CPU sort/merge
    Tim.start();
    mergeSmallBatch_CPU(h_A, h_B, n_A, n_B, n_batch, h_M);
    Tim.add();
    float cpu_time_merge = Tim.getsum();
    verify_result(h_M, d, n_batch);
    printf("\nCPU Merge completed in %f s.\n", cpu_time_merge);

    // GPU sort/merge
    int *d_A, *d_B, *d_M;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, (n_batch * n_A) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (n_batch * n_B) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, (n_batch * d) * sizeof(int)));
    
    // Copying arrays from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (n_batch * n_A) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (n_batch * n_B) * sizeof(int), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = (1024 / d) * d;
    int groupesPerBlock  = threadsPerBlock / d;
    int numBlocks       = (n_batch + groupesPerBlock - 1) / groupesPerBlock;

    // Calculate the total bytes needed for the shared memory array
    int sharedMemBytes  = threadsPerBlock * sizeof(int);
    
    // WARM-UP RUN
    mergeSmallBatch_k <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_M, n_A, n_B, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TIMING KERNEL 1 (Shared Memory)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (n_batch * n_A) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (n_batch * n_B) * sizeof(int), cudaMemcpyHostToDevice));

    Tim.reset(); 
    Tim.start();
    mergeSmallBatch_k_optimized <<<numBlocks, threadsPerBlock, sharedMemBytes>>> (d_A, d_B, d_M, n_A, n_B, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Tim.add();
    float gpu_time_merge_shared = Tim.getsum();

    // TIMING KERNEL 2 (Global Memory)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (n_batch * n_A) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (n_batch * n_B) * sizeof(int), cudaMemcpyHostToDevice));

    Tim.reset(); 
    Tim.start();
    mergeSmallBatch_k <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_M, n_A, n_B, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    Tim.add();
    float gpu_time_merge = Tim.getsum();

    // Copy back to the host
    CUDA_CHECK(cudaMemcpy(h_M, d_M, (n_batch * d) * sizeof(int), cudaMemcpyDeviceToHost));
    Tim.add();
    float gpu_time_merge_to_cpu = Tim.getsum();

    // Check result
    verify_result(h_M, d, n_batch);

    // Printing results
    printf("\nGPU Merge SHARED completed in %f s.\n", gpu_time_merge_shared);
    printf("GPU Merge completed in %f s.\n", gpu_time_merge);
    printf("GPU Merge + transfer to CPU completed in %f s.\n", gpu_time_merge_to_cpu);

    // Free the personalized test memory
    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_M));
    delete[] h_A;
    delete[] h_B;
    delete[] h_M;


    // =====================================================================
    // PART 2: CONSTANT WORKLOAD BENCHMARK (Tailored Arrays)
    // =====================================================================
    int totalElems = 1 << 24;  // ~16.7 million elements
    int halfElems = totalElems / 2;

    printf("\n\n========== Constant Workload Benchmark ==========\n");
    printf("Allocating %d elements (~67 MB) on GPU...\n", totalElems);

    // Allocate Host and Device Memory ONCE (The total size never changes)
    h_A = new int[halfElems];
    h_B = new int[halfElems];
    CUDA_CHECK(cudaMalloc(&d_A, halfElems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, halfElems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_M, totalElems * sizeof(int)));

    printf("\n=== Execution Time vs Array Size (d) ===\n");
    printf("%-8s %-12s %-10s %-14s %-10s\n", "d", "N", "tpb", "groupes/bloc", "temps(ms)");
    printf("----------------------------------------------------------------\n");

    int ds[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int nd   = sizeof(ds) / sizeof(ds[0]);

    // WARMUP RUN for benchmark (using uninitialized memory just to wake up GPU)
    bench_with_timer(d_A, d_B, d_M, 512, 512, 1024, totalElems / 1024, Tim);

    for (int k = 0; k < nd; k++) {
        int d_bench  = ds[k];
        int sA_bench = d_bench / 2;
        int sB_bench = d_bench / 2;
        int N_bench  = totalElems / d_bench;  

        // 1. Generate completely fresh random data for this specific test
        std::generate(h_A, h_A + halfElems, []() { return rand() % 10000; });
        std::generate(h_B, h_B + halfElems, []() { return rand() % 10000; });

        // 2. Sort the arrays EXACTLY in chunks of the current sA and sB
        for (int i = 0; i < N_bench; i++) {
            std::sort(h_A + (i * sA_bench), h_A + ((i + 1) * sA_bench));
            std::sort(h_B + (i * sB_bench), h_B + ((i + 1) * sB_bench));
        }

        // 3. Copy the freshly prepared, custom data to the GPU
        CUDA_CHECK(cudaMemcpy(d_A, h_A, halfElems * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, halfElems * sizeof(int), cudaMemcpyHostToDevice));

        int tpb_bench            = (1024 / d_bench) * d_bench;
        int groupesParBloc_bench = tpb_bench / d_bench;

        // 4. Run the benchmark
        float ms = bench_with_timer(d_A, d_B, d_M, sA_bench, sB_bench, d_bench, N_bench, Tim);
        
        printf("%-8d %-12d %-10d %-14d %.3f\n", d_bench, N_bench, tpb_bench, groupesParBloc_bench, ms);
    }
    printf("----------------------------------------------------------------\n");

    // Final Cleanup
    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_M));
    delete[] h_A;
    delete[] h_B;

    return 0;
}