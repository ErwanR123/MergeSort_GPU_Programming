#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> // For std::sort
#include <cassert>
#include <cstdlib>   // For rand() and srand()
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

// Check result on the CPU
void verify_result(int* M, int n_M) {
  for (int i = 0; i < n_M - 1; i++) {
      assert(M[i] <= M[i+1]);
    }
}

// Merge two sorted arrays A and B in M.
// Thread of id i find M[i] using a binary search
// of the merge path on the diagonal {(x,y) | x + y = i}.
__global__ void mergeSmall_k(int* A, int* B,
                              int n_A, int n_B,
                              int* M) {

    // We assume num_block = 1
    int i = threadIdx.x;
    if (i >= n_A + n_B) return;

    // Allocate Shared Memory (n_A + n_B < 1024) 
    // In this case with n_block=1 and n_thread < 1024, using shared memory
    // make the code slower as copying the data take more time than what we gain.
    __shared__ int s_A[1024]; 
    __shared__ int s_B[1024];

    // Every thread performs exactly ONE global memory read.
    if (i < n_A) {
        s_A[i] = A[i];             
    } else {
        s_B[i - n_A] = B[i - n_A];
    }

    __syncthreads();

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

    // Q is on the merge path iff those 2 conditions are verified :
    //   (1) A[Qy] > B[Qx-1]  : B’s elements are correctly placed before A (not too far right)
    //   (2) A[Qy-1] <= B[Qx] : A’s elements are correctly placed before B (not too far down)
    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= n_B &&
            (Qy == n_A || Qx == 0 || s_A[Qy] > s_B[Qx - 1])) {

            if (Qx == n_B || Qy == 0 || s_A[Qy - 1] <= s_B[Qx]) {
                // Q is on the merge path, we can now decide the value of M[i]
                if (Qy < n_A && (Qx == n_B || s_A[Qy] <= s_B[Qx])){
                    M[i] = s_A[Qy];
                } else {
                    M[i] = s_B[Qx];
                }
                return;
            } else {
                // Q is too far down, and need to be higher (hence more on the right) 
                // We update the low point of the diagonal to keep only the top right part
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            // Q is too far right, and need to be more on the left (hence lower) 
            // We update the high point of the diagonal to keep only the bottom left part
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

int main() {

    Timer Tim;
    int n_A, n_B, n_M;

    printf("\n========== Hand made test ==========\n");
    printf("Number of elements in A (<1024): ");
    std::cin >> n_A;
    printf("Number of elements in B (<1024 - n_A): ");
    std::cin >> n_B;

    n_M = n_A + n_B;
    assert(n_M <= 1024);
    
    int* h_A = new int[n_A];
    int* h_B = new int[n_B];
    int* h_M = new int[n_A + n_B];

    // Generate random arrays
    generate(h_A, h_A + n_A, []() { return rand() % 1000; });
    generate(h_B, h_B + n_B, []() { return rand() % 1000; });

    // 2. Sort the arrays BEFORE merging them
    std::sort(h_A, h_A + n_A);
    std::sort(h_B, h_B + n_B);
    
    // CPU sort/merge
    Tim.start();
    mergeCPU(h_A, n_A, h_B, n_B, h_M);
    Tim.add();
    float cpu_time_merge = Tim.getsum();
    verify_result(h_M, n_M);
    printf("CPU Merge completed in %f s.\n", cpu_time_merge);

    // GPU sort/merge
    int *d_A, *d_B, *d_M;

    CUDA_CHECK(cudaMalloc((void**)&d_A, n_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, n_M * sizeof(int)));

    // Copying arrays from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n_A * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n_B * sizeof(int), cudaMemcpyHostToDevice));

    Tim.reset(); Tim.start();
        
    mergeSmall_k <<<1, n_M>>> (d_A, d_B, n_A, n_B, d_M);
    CUDA_CHECK(cudaDeviceSynchronize());

    Tim.add();
    float gpu_time_merge = Tim.getsum();

    // Copy back to the host
    CUDA_CHECK(cudaMemcpy(h_M, d_M, n_M * sizeof(int), cudaMemcpyDeviceToHost));
    Tim.add();
    float gpu_time_merge_to_cpu = Tim.getsum();

    // Check result
    verify_result(h_M, n_M);

    // Printing results
    printf("\nGPU Merge completed in %f s.\n", gpu_time_merge);
    printf("GPU Merge + transfer to CPU completed in %f s.\n", gpu_time_merge_to_cpu);

    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_M));

    // FIXED: Prevent memory leaks
    delete[] h_A;
    delete[] h_B;
    delete[] h_M;

    return 0;
}
