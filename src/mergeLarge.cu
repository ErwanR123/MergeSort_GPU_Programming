#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cassert>
#include <thread>
#include <chrono>

// Include your custom timer
#include "timer.h" 

void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Renamed macro to avoid naming conflicts with the function
#define CUDA_CHECK(error) (testCUDA(error, __FILE__ , __LINE__))

void mergeCPU(int* A, int n_A,
                int* B, int n_B,
                int* M){
    int i = 0;
    int j = 0;

    while (i+j < n_A + n_B)
    {
        if (i >= n_A){
            M[i+j] = B[j];
            j += 1;
        }
        else if (j >= n_B)
        {
            M[i+j] = A[i];
            i += 1;
        }
        else if (A[i] < B[j])
        {
            M[i+j] = A[i];
            i += 1;
        }
        else
        {
            M[i+j] = B[j];
            j += 1;
        }
    }    
}

// The naive approch
__global__ void naive_mergeLarge_k(int* A, int n_A, int* B, int n_B, int* M) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_A + n_B) return;

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
                if (Qy < n_A && (Qx == n_B || A[Qy] <= B[Qx])) {
                    M[i] = A[Qy];
                } else {
                    M[i] = B[Qx];
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

// THE 2-STAGE APPROACH: STAGE 1 (Partitioning )
__global__ void merge_partitions_k(int* A, int n_A, int* B, int n_B, 
                                   int* bounds_A, int* bounds_B, int elements_per_block) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = n_A + n_B;
    int num_partitions = (total_elements + elements_per_block - 1) / elements_per_block + 1;

    // Only need one thread per block boundary
    if (i >= num_partitions) return;

    int diag = min(i * elements_per_block, total_elements);

    // Run Merge Path to find the exact boundary on this diagonal
    int Kx, Ky, Px, Py;
    if (diag > n_A) {
        Kx = diag - n_A;  Ky = n_A;
        Px = n_A;         Py = diag - n_A;
    } else {
        Kx = 0;           Ky = diag;
        Px = diag;        Py = 0;
    }

    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= n_B && (Qy == n_A || Qx == 0 || A[Qy] > B[Qx - 1])) {
            if (Qx == n_B || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                bounds_A[i] = Qy;
                bounds_B[i] = Qx;
                return;
            } else {
                Kx = Qx + 1; Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1; Py = Qy + 1;
        }
    }
}

// THE 2-STAGE APPROACH: STAGE 2 (Binary search on each block)
__global__ void mergeLarge_stage2_k(int* A, int n_A, int* B, int n_B, int* M, 
                                    int* bounds_A, int* bounds_B, int elements_per_block) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int total_elements = n_A + n_B;
    int diag = min(bx * elements_per_block, total_elements);
    int next_diag = min((bx + 1) * elements_per_block, total_elements);
    int block_size = next_diag - diag;

    if (block_size <= 0) return;

    // Read the exact starting and ending indices
    int start_A = bounds_A[bx];
    int start_B = bounds_B[bx];
    int local_n_A = bounds_A[bx + 1] - start_A;
    int local_n_B = bounds_B[bx + 1] - start_B;

    extern __shared__ int s_data[];
    int* s_A = s_data;
    int* s_B = s_data + local_n_A;

    // Load A and B chunck on the shared memory
    if (tx < local_n_A) {
        s_A[tx] = A[start_A + tx];
    } else if (tx < block_size) {
        s_B[tx - local_n_A] = B[start_B + (tx - local_n_A)];
    }

    __syncthreads();

    if (tx >= block_size) return;

    // Binary seach only on the tiny shared memory arrays
    int i = tx;
    int Kx, Ky, Px, Py;
    if (i > local_n_A) {
        Kx = i - local_n_A;  Ky = local_n_A;
        Px = local_n_A;      Py = i - local_n_A;
    } else {
        Kx = 0;              Ky = i;
        Px = i;              Py = 0;
    }

    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= local_n_B && (Qy == local_n_A || Qx == 0 || s_A[Qy] > s_B[Qx - 1])) {
            if (Qx == local_n_B || Qy == 0 || s_A[Qy - 1] <= s_B[Qx]) {
                if (Qy < local_n_A && (Qx == local_n_B || s_A[Qy] <= s_B[Qx])) {
                    M[diag + i] = s_A[Qy];
                } else {
                    M[diag + i] = s_B[Qx];
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

// Check result on the CPU
void verify_result(int* M, int n_M) {
  for (int i = 0; i < n_M - 1; i++) {
      assert(M[i] <= M[i+1]);
    }
}

int main() {
    Timer Tim;

    printf("\nInitializing GPU and running Warmup...\n");
    int warmup_d = 1024;
    int* d_W_A; int* d_W_B; int* d_W_M; int* d_W_bounds_A; int* d_W_bounds_B;
    CUDA_CHECK(cudaMalloc(&d_W_A, (warmup_d / 2) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W_B, (warmup_d / 2) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W_M, warmup_d * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W_bounds_A, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_W_bounds_B, 2 * sizeof(int)));
    
    // Fire the kernels once to wake up the GPU scheduler and L2 Cache
    naive_mergeLarge_k<<<1, 1024>>>(d_W_A, warmup_d/2, d_W_B, warmup_d/2, d_W_M);
    merge_partitions_k<<<1, 256>>>(d_W_A, warmup_d/2, d_W_B, warmup_d/2, d_W_bounds_A, d_W_bounds_B, 1024);
    mergeLarge_stage2_k<<<1, 1024, 1024 * sizeof(int)>>>(d_W_A, warmup_d/2, d_W_B, warmup_d/2, d_W_M, d_W_bounds_A, d_W_bounds_B, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_W_A)); CUDA_CHECK(cudaFree(d_W_B)); CUDA_CHECK(cudaFree(d_W_M));
    CUDA_CHECK(cudaFree(d_W_bounds_A)); CUDA_CHECK(cudaFree(d_W_bounds_B));

    // =====================================================================
    // MAIN BENCHMARK
    // =====================================================================
    printf("\n=== Large Array Merge: Naive vs 2-Stage ===\n");
    printf("%-12s %-16s %-10s %-14s %-14s %-14s\n", "d (Total)", "Elements/Block", "Blocks", "CPU(ms)", "Naive(ms)", "2-Stage(ms)");
    printf("-----------------------------------------------------------------------\n");

    // Scale 'd'
    int ds[] = {1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24};
    int num_tests = sizeof(ds) / sizeof(ds[0]);

    for (int t = 0; t < num_tests; t++) {
        int d = ds[t];
        int n_A = d / 2;
        int n_B = d / 2;

        // Allocate CPU Memory & Generate Fresh Sorted Data
        int* h_A = new int[n_A];
        int* h_B = new int[n_B];
        int* h_M = new int[n_A + n_B];

        std::generate(h_A, h_A + n_A, []() { return rand() % 100000; });
        std::generate(h_B, h_B + n_B, []() { return rand() % 100000; });
        std::sort(h_A, h_A + n_A);
        std::sort(h_B, h_B + n_B);

        // Allocate and Copy
        int *d_A, *d_B, *d_M_naive, *d_M_2stage;
        CUDA_CHECK(cudaMalloc(&d_A, n_A * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_B, n_B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_M_naive, d * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_M_2stage, d * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, n_A * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, n_B * sizeof(int), cudaMemcpyHostToDevice));

        // CPU benchmark
        Tim.reset();
        Tim.start();
        mergeCPU(h_A, n_A, h_B, n_B, h_M);
        Tim.add();
        float cpu_time_merge = Tim.getsum() * 1000.0f;
        verify_result(h_M, n_A + n_B);
        // float cpu_time_merge = 1000.0f;

        // Letting hardware power states reset
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Naive approach
        int threadsPerBlock = 1024;
        int numBlocksNaive = (d + threadsPerBlock - 1) / threadsPerBlock;

        Tim.reset(); Tim.start();
        naive_mergeLarge_k<<<numBlocksNaive, threadsPerBlock>>>(d_A, n_A, d_B, n_B, d_M_naive);
        CUDA_CHECK(cudaDeviceSynchronize());
        Tim.add();
        float time_naive = Tim.getsum() * 1000.0f;
        CUDA_CHECK(cudaMemcpy(h_M, d_M_naive, (n_A + n_B) * sizeof(int), cudaMemcpyDeviceToHost));
        verify_result(h_M, n_A + n_B);

        // Optimized approach
        int elements_per_block = 1024; // This dictates our Shared Memory chunk size
        int numBlocks2Stage = (d + elements_per_block - 1) / elements_per_block;
        int num_partitions = numBlocks2Stage + 1;

        // Allocate boundary arrays
        int *d_bounds_A, *d_bounds_B;
        CUDA_CHECK(cudaMalloc(&d_bounds_A, num_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bounds_B, num_partitions * sizeof(int)));

        Tim.reset(); Tim.start();
        
        // STAGE 1
        int numBlocksStage1 = (num_partitions + 255) / 256; 
        merge_partitions_k<<<numBlocksStage1, 256>>>(d_A, n_A, d_B, n_B, d_bounds_A, d_bounds_B, elements_per_block);
        
        // STAGE 2
        int sharedMemBytes = elements_per_block * sizeof(int);
        mergeLarge_stage2_k<<<numBlocks2Stage, elements_per_block, sharedMemBytes>>>(d_A, n_A, d_B, n_B, d_M_2stage, d_bounds_A, d_bounds_B, elements_per_block);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        Tim.add();
        float time_2stage = Tim.getsum() * 1000.0f;
        CUDA_CHECK(cudaMemcpy(h_M, d_M_2stage, (n_A + n_B) * sizeof(int), cudaMemcpyDeviceToHost));
        verify_result(h_M, n_A + n_B);

        printf("%-12d %-16d %-10d %-14.3f %-14.3f %-14.3f\n", d, elements_per_block, numBlocks2Stage, cpu_time_merge, time_naive, time_2stage);

        // Cleanup
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_M_naive)); CUDA_CHECK(cudaFree(d_M_2stage));
        CUDA_CHECK(cudaFree(d_bounds_A)); CUDA_CHECK(cudaFree(d_bounds_B));
        delete[] h_A; delete[] h_B;
    }
    
    printf("-----------------------------------------------------------------------\n");
    return 0;
}